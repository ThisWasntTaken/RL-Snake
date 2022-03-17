import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from torch.nn import Conv2d, MaxPool2d, Linear, Module

from agents import Double_DQN_Priority_Agent, Vanilla_DQN_Agent, Double_DQN_Agent
from utils import play, exponential_decay_schedule, linear_annealing_schedule, exponential_annealing_schedule, euclidean_distance
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
        high = 1,
        state_shape = (3, 50, 50)
        )
    agent = Double_DQN_Priority_Agent(
        environment = env,
        model_class = SnakeModel,
        learning_rate = 0.001,
        discount_factor = 0.999,
        epsilon_schedule = lambda n: exponential_decay_schedule(
            n = n,
            decay = 0.999,
            min_val = 1e-3
            ),
        beta_schedule = lambda n: exponential_annealing_schedule(
            n = n,
            rate = 1e-3
            ),
        replay_buffer_size = 50000,
        minimum_buffer_size = 5000,
        batch_size = 32,
        alpha=0.7,
        update_frequency = 4,
        device = torch.device('cpu')
    )
    rewards = agent.train(
        num_episodes = 20000,
        save_as = 'snake',
    )
    plt.plot(pd.Series(rewards).rolling(window=100).mean(), label = "Double DQN with Priority")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.legend()
    plt.title("Rolling average of 100 episode rewards")
    plt.tight_layout()
    plt.savefig("results/snake_rolling.png")
    # play(
    #     environment = env,
    #     model_class = SnakeModel,
    #     filepath = 'models/snake/8000.pth',
    #     num_episodes = 1
    # )