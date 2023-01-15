import gym
import gymnasium

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from config import Config
from models import DQN, ANFIS
from memory import Memory
from utils import wrap_input

LOSS_FNS = {
    "mse" : nn.MSELoss(),
    "huber" : nn.HuberLoss(),
}

OPTIMIZERS = {
    "Adam": optim.Adam,
}

ENVIRONMENTS = {
    # Classic control
    "CartPole-v1" : gym.make("CartPole-v1"),
    "Arcobot-v1" : gymnasium.make("Acrobot-v1"),
    "Pendulum-v1" : gym.make("Pendulum-v1"),
    "MountainCar-v0" : gym.make("MountainCar-v0"),

    # Box2d
    "LunarLander-v2" : gymnasium.make(
        "LunarLander-v2",
        continuous=False,
        gravity=-10.0,
        enable_wind=False,
        wind_power=15.0,
        turbulence_power=False,
    ),

    # Atari

    # External
}

def main() -> int:
    conf = Config("config.yaml")
    writer = SummaryWriter(f"./runs/{conf.training.env}/{conf.general.type}/")

    env = ENVIRONMENTS[conf.training.env]
    device = torch.device(conf.general.device if torch.cuda.is_available() else "cpu")

    # Online and offline model for learning
    if conf.general.type == "dqn":
        # DQN
        model = DQN(env.observation_space, env.action_space, conf.dqn.layer_size).to(device)
        target = DQN(env.observation_space, env.action_space, conf.dqn.layer_size).to(device)
    elif conf.general.type == "anfis":
        # ANFIS
        model = ANFIS(env.observation_space, env.action_space, conf.anfis.layer_size, conf.anfis.n_rules).to(device)
        target = ANFIS(env.observation_space, env.action_space, conf.anfis.layer_size, conf.anfis.n_rules).to(device)
    else:
        print(f"Model type is not implemented: {conf.general.type}")
        return -1

    # Optimizer and loss function
    optimizer = OPTIMIZERS[conf.optimizer.type](model.parameters(), lr=conf.optimizer.lr)
    loss_fn = LOSS_FNS[conf.loss.type]


    memory = Memory(conf.memory.max_len)
    obs, info = env.reset()

    for it in range(conf.training.n_iterations):
        # Do this for the batch norm
        model.eval()

        state = wrap_input(obs, device).unsqueeze(0)

        # Determine if we should take random action or greedy one
        if np.random.random() <= conf.training.epsilon:
            action = env.action_space.sample()
        else:
            action  = model(state).argmax().item()

        # Act in environment and store the memory
        next_state, reward, done, truncated, info = env.step(action)
        memory.store([obs, action, reward, done, next_state])

        if done:
            obs, info = env.reset()

        # Experience replay and training
        if it % conf.training.train_every == 0 and len(memory) > conf.training.batch_size:
            model.train()
            states, actions, rewards, dones, next_states = memory.sample(conf.training.batch_size)

            # Wrap and move all values to the cpu
            states = wrap_input(states, device)
            actions = wrap_input(actions, device, torch.int64)
            next_states = wrap_input(next_states, device)
            rewards = wrap_input(rewards, device)

            # Get current q-values
            qs = model(states)
            qs = qs.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Compute target q-values
            next_qs = target(next_states)
            next_qs = next_qs.max(dim=1)[0]
            next_qs[dones] = 0.0
            target_qs = rewards + conf.training.gamma * next_qs

            # Compute loss
            loss = loss_fn(qs, target_qs)
            writer.add_scalar("Loss", loss.item(), it)
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

            # soft update
            for target_param, local_param in zip(target.parameters(), model.parameters()):
                target_param.data.copy_(
                    conf.training.tau * local_param.data + (1 - conf.training.tau) * target_param.data
                )


        if it % conf.training.update_every == 0:
            target.load_state_dict(model.state_dict())

        if it % conf.testing.test_every == 0:
            print(f"{it:4}")
            with torch.no_grad():
                model.eval()
                rewards = []
                test_env = ENVIRONMENTS[conf.training.env]
                for episode in range(conf.testing.n_epiosdes):
                    done = False
                    obs, info = test_env.reset()
                    cum_reward = 0
                    while not done:
                        state = wrap_input(obs, device).unsqueeze(0)
                        action = model(state).argmax().item()
                        obs, reward, done, truncated, info = test_env.step(action)
                        cum_reward += reward

                    rewards.append(cum_reward)
                writer.add_scalar("Testing Mean Reward", np.mean(rewards), it)
                test_env.close()


    env.close()
    writer.close()

if __name__ == "__main__":
    raise SystemExit(main())
