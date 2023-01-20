import gym
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from models import DQN
from memory import Memory
from utils import wrap_input, epsilon_greedy

def main() -> int:
    env = gym.make("CartPole-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Online and offline model for learning
    model = DQN(env.observation_space, env.action_space, 24).to(device)
    target = DQN(env.observation_space, env.action_space, 24).to(device)
    target.eval()

    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=.001)
    loss_fn = F.smooth_l1_loss


    memory = Memory(10_000)
    obs, info = env.reset()

    for it in range(65_000):
        # Do this for the batch norm
        model.eval()

        # Maybe explore
        if np.random.random() <= epsilon_greedy(1.0, .01, 15_000, it):
            state = wrap_input(obs, device).unsqueeze(0)
            action  = model(state).argmax().item()
        else:
            action = env.action_space.sample()

        # Act in environment and store the memory
        next_state, reward, done, truncated, info = env.step(action)
        if truncated or done:
            next_state = np.zeros(env.observation_space.shape)
        memory.store([obs, action, reward, int(done), next_state])
        done = done or truncated

        if done:
            obs, info = env.reset()

        # Train
        if len(memory) > 32:
            model.train()
            states, actions, rewards, dones, next_states = memory.sample(32)

            # Wrap and move all values to the cpu
            states = wrap_input(states, device)
            actions = wrap_input(actions, device, torch.int64, reshape=True)
            next_states = wrap_input(next_states, device)
            rewards = wrap_input(rewards, device, reshape=True)
            dones = wrap_input(dones, device, reshape=True)

            # Get current q-values
            qs = model(states)
            qs = torch.gather(qs, dim=1, index=actions)

            # Compute target q-values
            with torch.no_grad():
                next_qs, _ = target(next_states).max(dim=1)
                next_qs = next_qs.reshape(-1, 1)

            target_qs = rewards + .9 * (1 - dones) * next_qs.reshape(-1, 1)

            # Compute loss
            loss = loss_fn(qs, target_qs)
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), 1)

            # Backprop
            optimizer.step()

            # soft update
            with torch.no_grad():
                for target_param, local_param in zip(target.parameters(), model.parameters()):
                    target_param.data.copy_(1e-2 * local_param.data + (1 - 1e-2) * target_param.data)


        if it % 200 == 0:
            target.load_state_dict(model.state_dict())


    model.load_state_dict(target.state_dict())
    # Test the model after training
    n_epiosdes = 10
    model.eval()
    rewards = []
    for episode in range(n_epiosdes):
        obs, info = env.reset()
        done = False
        counter = 0
        while not done:
            state = wrap_input(obs, device).unsqueeze(0)
            action = model(state).argmax().item()
            obs, reward, done, truncated, info = env.step(action)
            # done = done or truncated
            counter += 1
        rewards.append(counter)

    print(rewards)

    env.close()

if __name__ == "__main__":
    raise SystemExit(main())
