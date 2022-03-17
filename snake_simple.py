import torch
import matplotlib.pyplot as plt
import pandas as pd

from torch.nn import Linear, Module

from agents import Q_Agent, Vanilla_DQN_Agent, Double_DQN_Agent, Double_DQN_Priority_Agent
from utils import exponential_decay_schedule, linear_annealing_schedule, exponential_annealing_schedule
from snake import Snake


class SnakeModel(Module):
    def __init__(self):
        super(SnakeModel, self).__init__()
        self.fc1 = Linear(12, 32)
        self.fc2 = Linear(32, 16)
        self.fc3 = Linear(16, 4)
        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)
        torch.nn.init.kaiming_uniform_(self.fc3.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SnakeSimple(Snake):

    def __init__(self, num_columns, num_rows, low=0, high=1, state_shape=(12,), seed=None):
        super().__init__(
            num_columns = num_columns,
            num_rows = num_rows,
            low = low,
            high = high,
            state_shape = state_shape,
            seed = seed
            )
    
    @property
    def state(self):
        return (
            int((self.head.x + 1 > self.num_columns - 1) or ((self.head.x + 1, self.head.y) in self.body[1:])),
            int((self.head.x - 1 < 0) or ((self.head.x - 1, self.head.y) in self.body[1:])),
            int((self.head.y + 1 > self.num_rows - 1) or ((self.head.x, self.head.y + 1) in self.body[1:])),
            int((self.head.y - 1 < 0) or ((self.head.x, self.head.y - 1) in self.body[1:])),
            int(self.apple.x > self.head.x and self.apple.y == self.head.y),
            int(self.apple.y > self.head.y and self.apple.x == self.head.x),
            int(self.apple.x < self.head.x and self.apple.y == self.head.y),
            int(self.apple.y < self.head.y and self.apple.x == self.head.x),
            int(self.apple.x > self.head.x and self.apple.y > self.head.y),
            int(self.apple.x > self.head.x and self.apple.y < self.head.y),
            int(self.apple.x < self.head.x and self.apple.y > self.head.y),
            int(self.apple.x < self.head.x and self.apple.y < self.head.y),
            )
    
    def _crash(self):
        if self.head.x > self.num_columns - 1 or self.head.x < 0 or\
            self.head.y > self.num_rows - 1 or self.head.y < 0 or\
            self.head in self.body[1:]:
            return tuple([1] * 12), True

        return None, False


if __name__ == "__main__":
    env = SnakeSimple(
        num_columns = 20,
        num_rows = 20
        )
    agent = Q_Agent(
        environment=env,
        learning_rate=0.1,
        discount_factor=0.999,
        epsilon_schedule = lambda n: exponential_decay_schedule(
            n = n,
            decay = 0.999,
            min_val = 1e-3
            )
        )
    # rewards = agent.train(
    #     num_episodes = 20000,
    #     save_as = 'snake_q_table',
    # )
    # plt.plot(pd.Series(rewards).rolling(window=100).mean(), label = "Q Learning")
    # plt.xlabel("Episodes")
    # plt.ylabel("Rewards")
    # plt.legend()
    # plt.title("Rolling average of 100 episode rewards")
    # plt.tight_layout()
    # plt.savefig("results/snake_rolling.png")
    agent.play(
        filepath = 'models/snake_q_table.pkl',
        num_episodes = 1
    )
    # agent = Vanilla_DQN_Agent(
    #     environment = env,
    #     model_class = SnakeModel,
    #     learning_rate = 0.001,
    #     discount_factor = 0.999,
    #     epsilon_schedule = lambda n: exponential_decay_schedule(
    #         n = n,
    #         decay = 0.999,
    #         min_val = 0.01
    #         ),
    #     replay_buffer_size = 10000,
    #     minimum_buffer_size = 1000,
    #     batch_size = 32,
    #     update_frequency = 4,
    #     device = torch.device('cpu')
    # )
    # rewards = agent.train(
    #     num_episodes = 20000,
    #     save_as = 'snake',
    # )
    # plt.plot(pd.Series(rewards).rolling(window=100).mean(), label = "Vanilla DQN")
    # plt.xlabel("Episodes")
    # plt.ylabel("Rewards")
    # plt.legend()
    # plt.title("Rolling average of 100 episode rewards")
    # plt.tight_layout()
    # plt.savefig("results/snake_rolling.png")
    # agent.play(
    #     model_class = SnakeModel,
    #     filepath = 'models/snake/20000.pth',
    #     num_episodes = 1
    # )
    # agent = Double_DQN_Priority_Agent(
    #     environment = env,
    #     model_class = SnakeModel,
    #     learning_rate = 0.001,
    #     discount_factor = 0.999,
    #     epsilon_schedule = lambda n: exponential_decay_schedule(
    #         n = n,
    #         decay = 0.999,
    #         min_val = 1e-3
    #         ),
    #     beta_schedule = lambda n: exponential_annealing_schedule(
    #         n = n,
    #         rate = 1e-4
    #         ),
    #     replay_buffer_size = 50000,
    #     minimum_buffer_size = 50000,
    #     batch_size = 64,
    #     alpha=0.7,
    #     update_frequency = 4,
    #     device = torch.device('cpu')
    # )
    # rewards = agent.train(
    #     num_episodes = 20000,
    #     save_as = 'snake',
    # )
    # plt.plot(pd.Series(rewards).rolling(window=100).mean(), label = "Double DQN with Priority")
    # plt.xlabel("Episodes")
    # plt.ylabel("Rewards")
    # plt.legend()
    # plt.title("Rolling average of 100 episode rewards")
    # plt.tight_layout()
    # plt.savefig("results/snake_rolling.png")
    # agent.play(
    #     model_class = SnakeModel,
    #     filepath = 'models/7600.pth',
    #     num_episodes = 1
    # )