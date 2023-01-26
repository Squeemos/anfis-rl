import gym
import gymnasium

import numpy as np
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from config import Config
from models import DQN, ANFIS
from memory import Memory
from utils import wrap_input, make_env, epsilon_greedy

LOSS_FNS = {
    "mse" : nn.MSELoss(),
    "smooth_l1" : nn.SmoothL1Loss(),
}

OPTIMIZERS = {
    "Adam": optim.Adam,
}

def main() -> int:
    conf = Config("config.yaml")
    folder_path = f"{conf.training.env} - {conf.general.seed} - {conf.loss.type}"
    writer = SummaryWriter(f"./runs/{folder_path}/{conf.general.type}/")

    # Seed everything
    np.random.seed(conf.general.seed)
    torch.manual_seed(conf.general.seed)
    random.seed(conf.general.seed)

    env = make_env(conf.training.env)

    device = torch.device(conf.general.device if torch.cuda.is_available() else "cpu")

    # Online and offline model for learning
    if conf.general.type == "dqn":
        # DQN
        model = DQN(env.observation_space.shape, env.action_space.n, conf.dqn.layers).to(device)
        target = DQN(env.observation_space.shape, env.action_space.n, conf.dqn.layers).to(device)
    elif conf.general.type == "anfis":
        # ANFIS
        model = ANFIS(env.observation_space.shape, env.action_space.n, conf.anfis.layers, conf.anfis.n_rules, conf.anfis.defuzz_layers).to(device)
        target = ANFIS(env.observation_space.shape, env.action_space.n, conf.anfis.layers, conf.anfis.n_rules, conf.anfis.defuzz_layers).to(device)
    else:
        print(f"Model type is not implemented: {conf.general.type}")
        return -1

    print(model)

    # Don't let the target model train, saves some compute time
    target.eval()

    # Optimizer and loss function
    optimizer = OPTIMIZERS[conf.optimizer.type](model.parameters(), lr=conf.optimizer.lr)
    loss_fn = LOSS_FNS[conf.loss.type]


    memory = Memory(conf.memory.max_len)
    obs, info = env.reset()

    for it in range(0, conf.training.n_iterations + 1):
        if it % conf.training.log_every == 0:
            print(f"{it:7}", end="\r")

        # Do this for the batch norm
        model.eval()

        # Determine if we should take random action or greedy one
        epsilon = epsilon_greedy(conf.training.epsilon_start, conf.training.epsilon_end, conf.training.epsilon_n_decay, it)
        writer.add_scalar("Epsilon", epsilon, it)
        if np.random.random() <= epsilon:
            action = env.action_space.sample()
        else:
            state = wrap_input(obs, device).unsqueeze(0)
            action  = model(state).argmax().item()

        # Act in environment and store the memory
        next_state, reward, done, truncated, info = env.step(action)
        memory.store([obs, action, reward, int(done), next_state])

        obs = next_state
        done = done or truncated

        if done:
            obs, info = env.reset()

        # Experience replay and training
        if it >= conf.training.train_after:
            for _ in range(conf.training.gradient_steps):
                model.train()
                states, actions, rewards, dones, next_states = memory.sample(conf.training.batch_size)

                # Wrap and move all values to the gpu
                states = wrap_input(states, device)
                actions = wrap_input(actions, device, torch.int64, reshape=True)
                next_states = wrap_input(next_states, device)
                rewards = wrap_input(rewards, device, reshape=True)
                dones = wrap_input(dones, device, reshape=True)

                # Get current q-values
                qs = model(states)
                qs = torch.gather(qs, dim=1, index=actions)
                qs = qs.reshape(-1, 1)

                # Compute target q-values
                with torch.no_grad():
                    next_qs, _ = target(next_states).max(dim=1)
                    next_qs = next_qs.reshape(-1, 1)

                target_qs = rewards + conf.training.gamma * (1 - dones) * next_qs

                # Compute loss
                loss = loss_fn(qs, target_qs)
                writer.add_scalar("Loss", loss.item(), it)
                optimizer.zero_grad()
                loss.backward()

                # Clip gradients
                nn.utils.clip_grad_norm_(model.parameters(), conf.training.grad_norm)

                optimizer.step()

                # soft update
                with torch.no_grad():
                    for target_param, local_param in zip(target.parameters(), model.parameters()):
                        target_param.data.copy_(
                            conf.training.tau * local_param.data + (1 - conf.training.tau) * target_param.data
                        )

        # Update the double dqn
        if it % conf.training.update_every == 0:
            target.load_state_dict(model.state_dict())

        # Test the environment
        if it % conf.testing.test_every == 0:
            with torch.no_grad():
                model.eval()
                rewards = []
                # Make a copy so we can interact with that on our own without
                # messing up the training
                test_env  = make_env(conf.training.env)
                for episode in range(conf.testing.n_epiosdes):
                    done = False
                    obs, info = test_env.reset()
                    cum_reward = 0
                    while not done:
                        state = wrap_input(obs, device).unsqueeze(0)
                        action = model(state).argmax().item()
                        obs, reward, done, truncated, info = test_env.step(action)
                        cum_reward += reward
                        done = done or truncated

                    rewards.append(cum_reward)
                means = np.mean(rewards)
                medians = np.median(rewards)
                writer.add_scalar("Testing Mean Reward", means, it)
                writer.add_scalar("Testing Median Reward", medians, it)

                test_env.close()

                if means == 500 or medians == 500:
                    break

    print(f"Environment solved after {it} steps")


    env.close()
    writer.close()

if __name__ == "__main__":
    raise SystemExit(main())
