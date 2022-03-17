import gym
import torch
from torch.nn import Linear, Module

from agents import Vanilla_DQN_Agent
from utils import play

class MountainCarModel(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(2, 24)
        self.fc2 = Linear(24, 24)
        self.fc3 = Linear(24, 24)
        self.fc4 = Linear(24, 3)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    agent = Vanilla_DQN_Agent(
        environment = env,
        model_class = MountainCarModel,
        learning_rate = 0.001,
        discount_factor = 0.95,
        epsilon = 1,
        epsilon_decay = 0.995,
        epsilon_min = 0.01,
        replay_buffer_size = 50000,
        minimum_buffer_size = 5000,
        batch_size = 128,
        update_frequency = 500,
        device = torch.device('cpu')
    )
    agent.train(
        num_episodes = 500,
        save_as = 'mountaincar'
    )
    play(
        environment = env,
        model_class = MountainCarModel,
        filepath = 'models/mountaincar.pth',
        num_episodes = 1
    )