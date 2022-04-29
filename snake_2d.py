import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from torch.nn import Conv2d, MaxPool2d, Linear, Module

from agents import Double_DQN_Priority_Agent, Vanilla_DQN_Agent, Double_DQN_Agent
from utils import linear_decay_schedule, exponential_decay_schedule, linear_growth_schedule, exponential_growth_schedule
from snake import Snake


class SnakeModel(Module):
    def __init__(self):
        super(SnakeModel, self).__init__()
        self.conv1 = Conv2d(in_channels = 1, out_channels = 16, kernel_size = 5)
        # self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3)
        # self.maxpool2 = MaxPool2d(2)
        self.fc1 = Linear(32 * 4 * 4, 256)
        self.fc2 = Linear(256, 4)
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)
        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        # x = self.maxpool1(x)
        x = torch.relu(self.conv2(x))
        # x = self.maxpool2(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Snake2D(Snake):
    
    @property
    def state(self):
        state = np.zeros((self.num_columns, self.num_rows))
        state[self.apple.x][self.apple.y] = -1
        for i, j in self.body:
            state[i][j] = 1
        state[self.head.x][self.head.y] = 2
        return np.expand_dims(state, axis=0)


if __name__ == "__main__":
    env = Snake2D(
        num_columns = 10,
        num_rows = 10,
        low = -1,
        high = 2,
        state_shape = (1, 10, 10),
        seed=1242
        )
    # agent = Vanilla_DQN_Agent(
    #     environment = env,
    #     model_class = SnakeModel,
    #     learning_rate = 1e-3,
    #     discount_factor = 0.9,
    #     epsilon_schedule = lambda n: linear_decay_schedule(
    #         n = n,
    #         base = 1,
    #         rate = 5e-5,
    #         min_val = 1e-3
    #         ),
    #     replay_buffer_size = 50000,
    #     minimum_buffer_size = 10000,
    #     batch_size = 32,
    #     update_frequency = 4,
    #     device = torch.device('cpu')
    #     )
    # rewards, episode_lengths = agent.train(
    #     num_episodes = 50000,
    #     save_as = 'snake_2d_vanilla_dqn_linear_decay',
    #     )
    # plt.plot(pd.Series(rewards).rolling(window=100).mean(), label = "Reward")
    # plt.plot(pd.Series(episode_lengths).rolling(window=100).mean(), label = "Length")
    # plt.xticks([], [])
    # plt.legend()
    # plt.title("Rolling averages of 100 episodes")
    # plt.tight_layout()
    # plt.savefig("results/snake_2d_vanilla_dqn_linear_decay_rolling.png")
    # agent.play(
    #     model_class = SnakeModel,
    #     filepath = 'final/models/snake_2d_vanilla_dqn_linear_decay.pth',
    #     num_episodes = 1
    #     )
    agent = Double_DQN_Priority_Agent(
        environment = env,
        model_class = SnakeModel,
        learning_rate = 1e-3,
        discount_factor = 0.9,
        epsilon_schedule = lambda n: linear_decay_schedule(
            n = n,
            base = 1,
            rate = 5e-5,
            min_val = 1e-3
            ),
        beta_schedule = lambda n: linear_growth_schedule(
            n = n,
            base = 0.5,
            max_val = 1,
            rate = 1e-5
            ),
        replay_buffer_size = 50000,
        minimum_buffer_size = 10000,
        batch_size = 32,
        weight_decay = 1e-2,
        alpha=0.7,
        update_frequency = 4,
        device = torch.device('cpu'),
        seed=1242
        )
    # rewards, episode_lengths = agent.train(
    #     num_episodes = 50000,
    #     save_as = 'snake_2d_double_dqn_with_priority_linear_decay',
    #     )
    # plt.plot(pd.Series(rewards).rolling(window=100).mean(), label = "Reward")
    # plt.plot(pd.Series(episode_lengths).rolling(window=100).mean(), label = "Length")
    # plt.xlabel("Episodes")
    # plt.ylabel("Rewards")
    # plt.legend()
    # plt.title("Rolling average of 100 episode rewards")
    # plt.tight_layout()
    # plt.savefig("results/snake_2d_double_dqn_with_priority_linear_decay_rolling.png")
    agent.play(
        model_class = SnakeModel,
        filepath = 'final/models/snake_2d_double_dqn_with_priority_linear_decay.pth',
        num_episodes = 1
        )