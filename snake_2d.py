import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from torch.nn import Conv2d, MaxPool2d, Linear, Module

from agents import Double_DQN_Priority_Agent, Vanilla_DQN_Agent, Double_DQN_Agent
from utils import play, exponential_decay_schedule, linear_annealing_schedule, exponential_annealing_schedule
from snake import Snake


class SnakeModel(Module):
    def __init__(self):
        super(SnakeModel, self).__init__()
        self.conv1 = Conv2d(in_channels = 1, out_channels = 4, kernel_size = 3, padding = 1)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(in_channels = 4, out_channels = 8, kernel_size = 3, padding = 1)
        self.maxpool2 = MaxPool2d(2)
        self.fc1 = Linear(8 * 2 * 2, 8)
        self.fc2 = Linear(8, 4)
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)
        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = torch.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Snake2D(Snake):
    
    @property
    def state(self):
        state = np.zeros((self.num_columns, self.num_rows))
        state[self.apple.x][self.apple.y] = -2
        for i, j in self.body:
            state[i][j] = 1
        state[self.head.x][self.head.y] = 2
        return np.expand_dims(state, axis=0) / 2.0


if __name__ == "__main__":
    env = Snake2D(
        num_columns = 10,
        num_rows = 10,
        low = -1,
        high = 1,
        state_shape = (1, 10, 10)
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
        replay_buffer_size = 20000,
        minimum_buffer_size = 2000,
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
    # agent = Double_DQN_Priority_Agent(
    #     environment = env,
    #     model_class = SnakeModel,
    #     learning_rate = 0.001,
    #     discount_factor = 0.99,
    #     epsilon_schedule = lambda n: exponential_decay_schedule(
    #         n = n,
    #         decay = 0.9999,
    #         min_val = 1e-3
    #         ),
    #     beta_schedule = lambda n: exponential_annealing_schedule(
    #         n = n,
    #         rate = 1e-4
    #         ),
    #     replay_buffer_size = 500000,
    #     minimum_buffer_size = 10000,
    #     batch_size = 64,
    #     alpha=0.7,
    #     update_frequency = 4,
    #     device = torch.device('cpu')
    # )
    # rewards = agent.train(
    #     num_episodes = 5000,
    #     save_as = 'snake',
    # )
    # plt.plot(pd.Series(rewards).rolling(window=100).mean(), label = "Double DQN with Priority")
    # plt.xlabel("Episodes")
    # plt.ylabel("Rewards")
    # plt.legend()
    # plt.title("Rolling average of 100 episode rewards")
    # plt.tight_layout()
    # plt.savefig("results/snake.png")
    # play(
    #     environment = env,
    #     model_class = SnakeModel,
    #     filepath = 'models/snake/5000.pth',
    #     num_episodes = 1
    # )