import torch
import matplotlib.pyplot as plt
import pandas as pd

from torch.nn import Linear, Module

from agents import Q_Agent, Vanilla_DQN_Agent, Double_DQN_Agent, Double_DQN_Priority_Agent
from utils import linear_decay_schedule, linear_decay_schedule_after_k, exponential_decay_schedule, linear_growth_schedule, exponential_growth_schedule
from snake import Snake, Position


class SnakeModel(Module):
    def __init__(self):
        super(SnakeModel, self).__init__()
        self.fc1 = Linear(16, 32)
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

    def __init__(self, num_columns, num_rows, low=0, high=1, state_shape=(16,), seed=None):
        super().__init__(
            num_columns = num_columns,
            num_rows = num_rows,
            low = low,
            high = high,
            state_shape = state_shape,
            seed = seed
            )
    
    def _blocked(self, x, y):
        return not ((0 <= x < self.num_columns and 0 <= y < self.num_rows) and (Position(x, y) not in self.body))
    
    @property
    def state(self):
        return (
            int(self._blocked(self.head.x + 1, self.head.y)),
            int(self._blocked(self.head.x - 1, self.head.y)),
            int(self._blocked(self.head.x, self.head.y + 1)),
            int(self._blocked(self.head.x, self.head.y - 1)),
            int(self._blocked(self.head.x + 1, self.head.y + 1)),
            int(self._blocked(self.head.x + 1, self.head.y - 1)),
            int(self._blocked(self.head.x - 1, self.head.y + 1)),
            int(self._blocked(self.head.x - 1, self.head.y - 1)),
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
            return tuple([1 for _ in range(16)]), True

        return None, False


if __name__ == "__main__":
    env = SnakeSimple(
        num_columns = 10,
        num_rows = 10
        )
    agent = Q_Agent(
        environment=env,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon_schedule = lambda n: linear_decay_schedule(
            n = n,
            base = 1,
            rate = 1e-4,
            min_val = 1e-3
            )
        )
    rewards, episode_lengths = agent.train(
        num_episodes = 20000,
        save_as = 'snake_simple_q_table_linear_decay',
        )
    plt.plot(pd.Series(rewards).rolling(window=100).mean(), label = "Reward")
    plt.plot(pd.Series(episode_lengths).rolling(window=100).mean(), label = "Length")
    plt.legend()
    plt.title("Rolling averages of 100 episodes")
    plt.tight_layout()
    plt.savefig("results/snake_simple_q_table_linear_decay_rolling.png")
    agent.play(
        filepath = 'models/snake_simple_q_table_linear_decay.pkl',
        num_episodes = 1
        )
    # agent = Vanilla_DQN_Agent(
    #     environment = env,
    #     model_class = SnakeModel,
    #     learning_rate = 0.001,
    #     discount_factor = 0.9,
    #     epsilon_schedule = lambda n: linear_decay_schedule(
    #         n = n,
    #         base = 1,
    #         rate = 1e-4,
    #         min_val = 1e-3
    #         ),
    #     replay_buffer_size = 10000,
    #     minimum_buffer_size = 1000,
    #     batch_size = 32,
    #     weight_decay = 1e-2,
    #     update_frequency = 4,
    #     device = torch.device('cpu')
    #     )
    # rewards, episode_lengths = agent.train(
    #     num_episodes = 20000,
    #     save_as = 'snake_simple_vanilla_dqn_linear_decay',
    #     )
    # plt.plot(pd.Series(rewards).rolling(window=100).mean(), label = "Reward")
    # plt.plot(pd.Series(episode_lengths).rolling(window=100).mean(), label = "Length")
    # plt.legend()
    # plt.title("Rolling averages of 100 episodes")
    # plt.tight_layout()
    # plt.savefig("results/snake_simple_vanilla_dqn_linear_decay_rolling.png")
    # agent.play(
    #     model_class = SnakeModel,
    #     filepath = 'models/snake_simple_vanilla_dqn_linear_decay/20000.pth',
    #     num_episodes = 1
    #     )
    # agent = Double_DQN_Priority_Agent(
    #     environment = env,
    #     model_class = SnakeModel,
    #     learning_rate = 0.001,
    #     discount_factor = 0.9,
    #     epsilon_schedule = lambda n: linear_decay_schedule(
    #         n = n,
    #         base = 1,
    #         rate = 1e-4,
    #         min_val = 1e-3
    #         ),
    #     beta_schedule = lambda n: linear_growth_schedule(
    #         n = n,
    #         base = 0.5,
    #         max_val = 1,
    #         rate = 2e-5
    #         ),
    #     replay_buffer_size = 10000,
    #     minimum_buffer_size = 1000,
    #     batch_size = 32,
    #     weight_decay = 1e-2,
    #     alpha = 0.7,
    #     update_frequency = 4,
    #     device = torch.device('cpu')
    #     )
    # rewards, episode_lengths = agent.train(
    #     num_episodes = 20000,
    #     save_as = 'snake_simple_double_dqn_with_priority_linear_decay',
    #     )
    # plt.plot(pd.Series(rewards).rolling(window=100).mean(), label = "reward")
    # plt.plot(pd.Series(episode_lengths).rolling(window=100).mean(), label = "Length")
    # plt.xlabel("Episodes")
    # plt.ylabel("Rewards")
    # plt.legend()
    # plt.title("Rolling average of 100 episode rewards")
    # plt.tight_layout()
    # plt.savefig("results/snake_simple_double_dqn_with_priority_linear_decay_rolling.png")
    # agent.play(
    #     model_class = SnakeModel,
    #     filepath = 'models/snake_simple_double_dqn_with_priority_linear_decay/20000.pth',
    #     num_episodes = 1
    #     )