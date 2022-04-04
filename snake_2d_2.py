import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from torch.nn import Conv2d, MaxPool2d, Linear, Module

from agents import Double_DQN_Priority_Agent, Vanilla_DQN_Agent, Double_DQN_Agent
from utils import exponential_decay_schedule, linear_growth_schedule, exponential_growth_schedule
from snake import Snake, Position


class SnakeModel(Module):
    def __init__(self):
        super(SnakeModel, self).__init__()
        self.conv1 = Conv2d(in_channels = 1, out_channels = 4, kernel_size = 3)
        self.conv2 = Conv2d(in_channels = 4, out_channels = 8, kernel_size = 3)
        self.fc1 = Linear(8 * 3 * 3, 64)
        self.fc2 = Linear(64, 4)
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)
        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Snake2D(Snake):
    
    def _blocked(self, x, y):
        return not ((0 <= x < self.num_columns and 0 <= y < self.num_rows) and (Position(x, y) not in self.body))
    
    @property
    def state(self):
        state = np.zeros((7, 7))
        for i, x in enumerate(range(self.head.x - 3, self.head.x + 4)):
            for j, y in enumerate(range(self.head.y - 3, self.head.y + 4)):
                if self._blocked(x, y):
                    state[i][j] = 1
                elif Position(x, y) == self.apple:
                    state[i][j] = -1
        return np.expand_dims(state, axis=0)


if __name__ == "__main__":
    env = Snake2D(
        num_columns = 10,
        num_rows = 10,
        low = -1,
        high = 1,
        state_shape = (1, 7, 7)
        )
    agent = Vanilla_DQN_Agent(
        environment = env,
        model_class = SnakeModel,
        learning_rate = 0.01,
        discount_factor = 0.999,
        epsilon_schedule = lambda n: exponential_decay_schedule(
            n = n,
            decay = 0.999,
            min_val = 1e-3
            ),
        replay_buffer_size = 20000,
        minimum_buffer_size = 2000,
        batch_size = 32,
        update_frequency = 4,
        device = torch.device('cpu')
    )
    rewards, episode_lengths = agent.train(
        num_episodes = 20000,
        save_as = 'snake_2d_2_vanilla_dqn',
    )
    plt.plot(pd.Series(rewards).rolling(window=100).mean(), label = "Reward")
    plt.plot(pd.Series(episode_lengths).rolling(window=100).mean(), label = "Length")
    plt.legend()
    plt.title("Rolling averages of 100 episodes")
    plt.tight_layout()
    plt.savefig("results/snake_2d_2_vanilla_dqn.png")
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
    #     beta_schedule = lambda n: exponential_growth_schedule(
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
    # agent.play(
    #     model_class = SnakeModel,
    #     filepath = 'models/15000.pth',
    #     num_episodes = 1
    # )