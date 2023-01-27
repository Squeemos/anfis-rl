import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import random

from .modules import DQN, ANFIS
from .utils import wrap_input, epsilon_greedy, make_env, soft_update
from .memory import Memory

class Agent(object):
    LOSS_FNS = {
        "mse" : nn.MSELoss(),
        "smooth_l1" : nn.SmoothL1Loss(),
    }

    OPTIMIZERS = {
        "adam": optim.Adam,
    }

    def __init__(
        self,
        model_type,
        env_id,
        layers,
        optimizer,
        lr,
        loss_fn,
        replay_buffer_size,
        device,
        seed=0,
        n_rules=16,
        defuzz_layers=[64, 64],
        writer=True,
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[Using {self.device}]")
        print(f"[Using {model_type} model]")
        print(f"[Setup for {env_id}]")

        self.env_id = env_id
        self.env = make_env(self.env_id)

        # Online and offline model for learning
        if model_type == "dqn":
            # DQN
            self.model = DQN(self.env.observation_space.shape, self.env.action_space.n, layers).to(self.device)
            self.target = DQN(self.env.observation_space.shape, self.env.action_space.n, layers).to(self.device)
        elif model_type == "anfis":
            # ANFIS
            self.model = ANFIS(self.env.observation_space.shape, self.env.action_space.n, layers, n_rules, defuzz_layers).to(self.device)
            self.target = ANFIS(self.env.observation_space.shape, self.env.action_space.n, layers, n_rules, defuzz_layers).to(self.device)
        else:
            print(f"Model type is not implemented: {conf.general.type}")
            return -1

        # Optimizer, loss function, and memory for experience replay
        self.optimizer = Agent.OPTIMIZERS[optimizer](self.model.parameters(), lr=lr)
        self.loss_fn = Agent.LOSS_FNS[loss_fn]
        self.memory = Memory(replay_buffer_size)

        # Use a tensorboard SummaryWriter
        if writer:
            folder_path = f"{env_id} - {seed} - {loss_fn}"
            self.writer = SummaryWriter(f"./runs/{folder_path}/{model_type}/")
        else:
            self.writer = None

        self.target.eval()

    def train(
        self,
        n_iterations,
        train_after,
        epsilon_start,
        epsilon_end,
        epsilon_decay,
        update_every,
        test_every,
        testing_episodes=10,
        batch_size=128,
        gamma=0.99,
        gradient_steps=1,
        tau=1e-3,
        grad_norm=10,
        print_updates=True,
    ):
        # Reset the env
        obs, info = self.env.reset()
        # For printing updates
        format_string = len(str(n_iterations + 1))

        for it in range(0, n_iterations + 1):
            if print_updates:
                print(f"Training Iteration: {it:{format_string}}", end="\r")

            # Do for batch norm and computation speed
            self.model.eval()

            # Epsilon greedy
            epsilon = epsilon_greedy(epsilon_start, epsilon_end, epsilon_decay * n_iterations, it)

            if self.writer is not None:
                self.writer.add_scalar("Epsilon", epsilon, it)

            # Maybe take a random action
            if np.random.random() <= epsilon:
                action = self.env.action_space.sample()
            else:
                state = wrap_input(obs, self.device).unsqueeze(0)
                action  = self.model(state).argmax(dim=-1).item()

            # Step in the environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            self.memory.store([obs, action, reward, int(terminated), next_state])

            # Prep for next iteration
            obs = next_state
            done = terminated or truncated
            if done:
                obs, info = self.env.reset()

            # Training
            if it >= train_after:
                for _ in range(gradient_steps):
                    self.model.train()
                    states, actions, rewards, dones, next_states = self.memory.sample(batch_size)

                    # Wrap and move all values to the gpu
                    states = wrap_input(states, self.device)
                    actions = wrap_input(actions, self.device, torch.int64, reshape=True)
                    next_states = wrap_input(next_states, self.device)
                    rewards = wrap_input(rewards, self.device, reshape=True)
                    dones = wrap_input(dones, self.device, reshape=True)

                    # Get current q-values
                    qs = self.model(states)
                    qs = torch.gather(qs, dim=1, index=actions)
                    qs = qs.reshape(-1, 1)

                    # Compute target q-values
                    with torch.no_grad():
                        next_qs, _ = self.target(next_states).max(dim=1)
                        next_qs = next_qs.reshape(-1, 1)

                    target_qs = rewards + gamma * (1 - dones) * next_qs

                    # Compute loss
                    loss = self.loss_fn(qs, target_qs)

                    if self.writer is not None:
                        self.writer.add_scalar("Loss", loss.item(), it)
                    self.optimizer.zero_grad()
                    loss.backward()

                    # Clip gradients
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_norm)

                    self.optimizer.step()

                    # soft update
                    soft_update(self.target, self.model, tau)

            # Update the double dqn
            if it % update_every == 0:
                self.target.load_state_dict(self.model.state_dict())

            # Test the environment
            if it % test_every == 0:
                with torch.no_grad():
                    self.model.eval()
                    rewards = []
                    # Make a copy so we can interact with that on our own without
                    # messing up the training
                    test_env  = make_env(self.env_id)
                    for episode in range(testing_episodes):
                        done = False
                        obs, info = test_env.reset()
                        cum_reward = 0
                        while not done:
                            state = wrap_input(obs, self.device).unsqueeze(0)
                            action = self.model(state).argmax(dim=-1).item()
                            obs, reward, terminated, truncated, info = test_env.step(action)
                            cum_reward += reward
                            done = terminated or truncated

                        rewards.append(cum_reward)
                    means = np.mean(rewards)
                    medians = np.median(rewards)
                    if self.writer is not None:
                        self.writer.add_scalar("Testing Mean Reward", means, it)
                        self.writer.add_scalar("Testing Median Reward", medians, it)

                    test_env.close()

        self.env.close()

    def shutdown(self):
        self.writer.close()
