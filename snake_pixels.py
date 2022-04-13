import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from torch.nn import Conv2d, MaxPool2d, Linear, Module

from agents import Double_DQN_Priority_Agent, Vanilla_DQN_Agent, Double_DQN_Agent
from utils import linear_decay_schedule, linear_decay_schedule_after_k, linear_growth_schedule, exponential_growth_schedule
from snake import Snake


class SnakeModel(Module):
    def __init__(self):
        super(SnakeModel, self).__init__()
        self.conv1 = Conv2d(in_channels = 3, out_channels = 16, kernel_size = 8, stride = 4)
        self.conv2 = Conv2d(in_channels = 16, out_channels = 32, kernel_size = 4, stride = 2)
        self.fc1 = Linear(32 * 4 * 4, 256)
        self.fc2 = Linear(256, 4)
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)
        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)

    def forward(self, x):
        # x = torch.permute(x, (0, 3, 1, 2))
        # print(x.shape)
        x = torch.relu(self.conv1(x))
        # print(x.shape)
        x = torch.relu(self.conv2(x))
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = torch.relu(self.fc1(x))
        # print(x.shape)
        x = self.fc2(x)
        # print(x.shape)
        return x


class SnakePixels(Snake):

    BLOCK_SIZE = 5

    def __init__(self, num_columns, num_rows, low=0, high=1, seed=None):
        super().__init__(
            num_columns = num_columns,
            num_rows = num_rows,
            low = low,
            high = high,
            state_shape = (3, num_columns * self.BLOCK_SIZE, num_rows * self.BLOCK_SIZE),
            seed = seed
            )
    
    @property
    def state(self):
        state = np.zeros((3, self.num_columns, self.num_rows))
        state[:, self.apple.x, self.apple.y] = (255, 0, 0)
        for i, j in self.body:
            state[:, i, j] = (0, 255, 0)
        state[:, self.head.x, self.head.y] = (255, 255, 255)
        state = np.repeat(np.repeat(state, self.BLOCK_SIZE, 2), self.BLOCK_SIZE, 1)
        return state / 255.0


if __name__ == "__main__":
    env = SnakePixels(
        num_columns = 10,
        num_rows = 10,
        low = 0,
        high = 1
        )
    agent = Double_DQN_Priority_Agent(
        environment = env,
        model_class = SnakeModel,
        learning_rate = 1e-2,
        discount_factor = 0.9,
        epsilon_schedule = lambda n: linear_decay_schedule(
            n = n,
            base = 1,
            rate = 1e-4,
            min_val = 1e-3
            ),
        beta_schedule = lambda n: linear_growth_schedule(
            n = n,
            base = 0.5,
            max_val = 1,
            rate = 2e-5
            ),
        replay_buffer_size = 20000,
        minimum_buffer_size = 2000,
        batch_size = 128,
        weight_decay = 1e-2,
        alpha=0.7,
        update_frequency = 4,
        device = torch.device('cuda:0')
        )
    rewards, episode_lengths = agent.train(
        num_episodes = 20000,
        save_as = 'snake_pixels_double_dqn_with_priority_linear_decay',
        )
    plt.plot(pd.Series(rewards).rolling(window=100).mean(), label = "Reward")
    plt.plot(pd.Series(episode_lengths).rolling(window=100).mean(), label = "Length")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.legend()
    plt.title("Rolling average of 100 episode rewards")
    plt.tight_layout()
    plt.savefig("results/snake_pixels_double_dqn_with_priority_linear_decay_rolling.png")
    agent.play(
        model_class = SnakeModel,
        filepath = 'models/snake_pixels_double_dqn_with_priority_linear_decay/20000.pth',
        num_episodes = 1
        )