import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

from torch.nn import functional as F, Module, init
from torch.nn.parameter import Parameter


def exponential_decay_schedule(n, decay, min_val):
    return max(decay ** n, min_val)


def linear_decay_schedule(n, base, rate, min_val):
    return max(base - n * rate, min_val)


def linear_decay_schedule_after_k(n, k, base, rate, min_val):
    if n <= k:
        return base
    
    return max(base - (n - k) * rate, min_val)


def linear_growth_schedule(n, base, rate, max_val):
    return min(max_val, base + n * rate)


def exponential_growth_schedule(n, rate):
    return 1 - np.exp(-rate * n)


def euclidean_distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))


def plot(num_episodes, rewards, episode_lengths, filepath):
    plt.plot(range(num_episodes), rewards, label = "Reward")
    plt.plot(range(num_episodes), episode_lengths, label = "Length")
    plt.xlabel("Episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/" + filepath + ".png")
    plt.cla()
    plt.plot(pd.Series(rewards).rolling(window=100).mean(), label = "Reward")
    plt.plot(pd.Series(episode_lengths).rolling(window=100).mean(), label = "Length")
    plt.legend()
    plt.title("Rolling averages of 100 episodes")
    plt.tight_layout()
    plt.savefig("results/" + filepath + "_rolling.png")
    plt.cla()


class ExperienceReplayBuffer:
    def __init__(self, capacity, state_shape, batch_size, rng):
        self.capacity = capacity
        self.batch_size = batch_size
        self.states = np.zeros(((capacity,) + state_shape), dtype=np.float64)
        self.actions = torch.zeros((capacity), dtype = torch.long)
        self.new_states = np.zeros(((capacity,) + state_shape), dtype=np.float64)
        self.rewards = torch.zeros((capacity))
        self.mask = torch.ones((capacity), dtype = torch.bool)
        self.position = 0
        self.size = 0
        self.rng = rng
    
    def push(self, state, action, reward, new_state, done, _pos=None):
        if _pos is None:
            _pos = self.position
        
        self.states[_pos] = np.array(state)
        self.actions[_pos] = action
        self.rewards[_pos] = reward
        self.new_states[_pos] = np.array(new_state)
        self.mask[_pos] = 0 if done else 1
            
        self.position = (self.position + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
    
    def sample(self):
        indices = self.rng.choice(
            a = self.size,
            size = (self.batch_size,),
            replace=True
            )
        return [
            torch.from_numpy(self.states[indices]),
            self.actions[indices],
            self.rewards[indices],
            torch.from_numpy(self.new_states[indices]),
            self.mask[indices],
            ]

    def empty(self):
        return self.size == 0
    
    def full(self):
        return self.size == self.capacity
    
    def __len__(self):
        return self.size


class PrioritizedExperienceReplayBuffer(ExperienceReplayBuffer):
    def __init__(self, capacity, state_shape, batch_size, alpha, rng):
        super().__init__(
            capacity = capacity,
            state_shape = state_shape,
            batch_size = batch_size,
            rng = rng
            )
        self.priorities = np.zeros((capacity), dtype=np.float64)
        self.alpha = alpha
    
    def push(self, state, action, reward, new_state, done):
        priority = 1.0 if self.empty() else self.priorities.max()
        if not self.full():
            self.priorities[self.position] = priority
            super().push(state, action, reward, new_state, done)
        else:
            idx = self.priorities.argmin()
            if priority > self.priorities[idx]:
                self.priorities[idx] = priority
                super().push(state, action, reward, new_state, done, idx)
    
    def sample(self, beta):
        s = (self.priorities ** self.alpha).sum()
        p = (self.priorities ** self.alpha) / s
        indices = self.rng.choice(
            a = self.capacity,
            size = (self.batch_size,),
            replace=True,
            p=p
            )
        weights = (self.size * p[indices]) ** (-beta)
        normalized_weights = weights / weights.max()
        return [
            torch.from_numpy(self.states[indices]),
            self.actions[indices],
            self.rewards[indices],
            torch.from_numpy(self.new_states[indices]),
            self.mask[indices],
            indices,
            torch.from_numpy(normalized_weights)
            ]
    
    def update(self, indices, priorities):
        self.priorities[indices] = priorities


class NoisyNormalLinear(Module):
    def __init__(self, in_features, out_features, mean=0, std=1, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.weight_noise = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            self.bias_noise = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('bias_noise', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.normal_(self.weight_noise, self.mean, self.std)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            init.normal_(self.bias_noise, self.mean, self.std)
    
    def forward(self, input):
        if not self.training:
            return F.linear(input, self.weight, self.bias)
        return F.linear(
            input,
            self.weight + self.weight_noise,
            (self.bias + self.bias_noise) if self.bias is not None else self.bias
        )
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, mean={}, std={}, bias={}'.format(
            self.in_features, self.out_features, self.mean, self.std, self.bias is not None
        )