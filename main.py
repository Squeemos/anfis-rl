import gym
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from models import DQN
from memory import Memory

def wrap_input(arr, device, dtype=torch.float):
    return torch.from_numpy(np.array(arr)).type(dtype).to(device)

def main() -> int:
    env = gym.make("CartPole-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Online and offline model for learning
    model = DQN(env.observation_space, env.action_space).to(device)
    target = DQN(env.observation_space, env.action_space).to(device)

    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=.001)
    loss_fn = nn.MSELoss()


    memory = Memory(10_000)
    obs = env.reset()

    for it in range(1_000):
        # Do this for the bath norm
        model.eval()

        # Take the action the network tells us to
        # TODO: environment exploration
        state = wrap_input(obs, device).unsqueeze(0)
        action  = model(state).argmax().item()

        # Act in environment and store the memory
        next_state, reward, done, info = env.step(action)
        memory.store([obs, action, reward, done, next_state])

        if done:
            obs = env.reset()

        if len(memory) > 128:
            model.train()
            states, actions, rewards, dones, next_states = memory.sample(128)

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
            target_qs = rewards + 0.5 * next_qs

            # Compute loss
            loss = loss_fn(qs, target_qs)
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

            # soft update
            for target_param, local_param in zip(target.parameters(), model.parameters()):
                target_param.data.copy_(1e-3 * local_param.data + (1 - 1e-3) * target_param.data)


        if it % 8 == 0:
            target.load_state_dict(model.state_dict())


    # Test the model after training
    n_epiosdes = 10
    model.eval()
    rewards = []
    for episode in range(n_epiosdes):
        obs = env.reset()
        done = False
        counter = 0
        while not done:
            state = wrap_input(obs, device).unsqueeze(0)
            action = model(state).argmax().item()
            obs, reward, done, info = env.step(action)
            counter += 1
            env.render()
        rewards.append(counter)

    print(rewards)

    env.close()

if __name__ == "__main__":
    raise SystemExit(main())
