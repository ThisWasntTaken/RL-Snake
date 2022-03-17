import gym
import torch
import pandas as pd
import matplotlib.pyplot as plt

from torch.nn import Linear, Module
from agents import *
from utils import exponential_annealing_schedule, play, exponential_decay_schedule, linear_annealing_schedule


class CartPoleModel(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(4, 24)
        self.fc2 = Linear(24, 24)
        self.fc3 = Linear(24, 2)
        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)
        torch.nn.init.kaiming_uniform_(self.fc3.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = Vanilla_DQN_Agent(
        environment = env,
        model_class = CartPoleModel,
        learning_rate = 0.001,
        discount_factor = 0.95,
        epsilon_schedule = lambda n: exponential_decay_schedule(
            n = n,
            decay = 0.995,
            min_val = 0.1
            ),
        replay_buffer_size = 10000,
        minimum_buffer_size = 500,
        batch_size = 64,
        update_frequency = 4,
        device = torch.device('cpu'),
        seed = 1234
    )
    rewards = agent.train(
        num_episodes = 1000,
        save_as = 'cartpole',
    )
    plt.plot(pd.Series(rewards).rolling(window=100).mean(), label = "Vanilla DQN")
    agent = Double_DQN_Agent(
        environment = env,
        model_class = CartPoleModel,
        learning_rate = 0.001,
        discount_factor = 0.95,
        epsilon_schedule = lambda n: exponential_decay_schedule(
            n = n,
            decay = 0.995,
            min_val = 0.1
            ),
        replay_buffer_size = 10000,
        minimum_buffer_size = 1000,
        batch_size = 64,
        update_frequency = 3,
        device = torch.device('cpu'),
        seed = 1234
    )
    rewards = agent.train(
        num_episodes = 1000,
        save_as = 'cartpole',
    )
    plt.plot(pd.Series(rewards).rolling(window=100).mean(), label = "Double DQN")
    agent = Double_DQN_Priority_Agent(
        environment = env,
        model_class = CartPoleModel,
        learning_rate = 0.001,
        discount_factor = 0.95,
        epsilon_schedule = lambda n: exponential_decay_schedule(
            n = n,
            decay = 0.995,
            min_val = 0.1
            ),
        beta_schedule = lambda n: exponential_annealing_schedule(
            n = n,
            rate = 1e-3
            ),
        replay_buffer_size = 10000,
        minimum_buffer_size = 1000,
        batch_size = 64,
        alpha=0.7,
        update_frequency = 3,
        device = torch.device('cpu'),
        seed = 1234
    )
    rewards = agent.train(
        num_episodes = 1000,
        save_as = 'cartpole',
    )
    plt.plot(pd.Series(rewards).rolling(window=100).mean(), label = "Double DQN with Priority")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.legend()
    plt.title("Rolling average of 100 episode rewards")
    plt.tight_layout()
    plt.savefig("results/comparison.png")
    # play(
    #     environment = env,
    #     model_class = CartPoleModel,
    #     filepath = 'models/cartpole.pth',
    #     num_episodes = 1
    # )