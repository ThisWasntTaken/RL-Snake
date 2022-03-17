import torch
import tqdm
import numpy as np
import pickle

from utils import ExperienceReplayBuffer, PrioritizedExperienceReplayBuffer, plot


SHOW_EVERY = 0


class Q_Agent:
    def __init__(self, environment, learning_rate, discount_factor=0.999, epsilon_schedule=lambda: 0.9, seed=None):
        self.env = environment
        self.q_table = dict()
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_schedule = epsilon_schedule
        self.seed(seed)
    
    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed) if seed else np.random.default_rng()
    
    def get_action(self, state, epsilon):
        if state not in self.q_table:
            self.q_table[state] = np.zeros((self.env.action_space.n,), dtype=np.float32)
        
        gen = self.rng.uniform(0, 1)
        if gen < epsilon:
            return self.env.action_space.sample()
        
        return np.argmax(self.q_table[state])
    
    def get_max_q_value(self, state):
        if state in self.q_table:
            return np.max(self.q_table[state])
        
        # self.q_table[state] = np.zeros((self.env.action_space.n,), dtype=np.float32)
        return 0.0
    
    def get_q_value(self, state, action):
        if state in self.q_table:
            return self.q_table[state][action]
        
        # self.q_table[state] = np.zeros((self.env.action_space.n,), dtype=np.float32)
        return 0.0
    
    def train(self, num_episodes, save_as):
        rewards_list = []

        for e in tqdm.tqdm(range(1, num_episodes + 1)):
            state = self.env.reset()
            done = False
            episode_reward = 0
            epsilon = self.epsilon_schedule(e)
            while not done:
                action = self.get_action(state, epsilon)
                if SHOW_EVERY and e % SHOW_EVERY == 0:
                    self.env.render()
                new_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                q_value = self.get_q_value(state, action)
                max_q_value = self.get_max_q_value(new_state)
                self.q_table[state][action] = (1 - self.learning_rate) * q_value + self.learning_rate * (reward + self.discount_factor * max_q_value)
                state = new_state
            
            rewards_list.append(episode_reward)
        
        checkpoint = {"q_table": self.q_table}
        self.save(checkpoint = checkpoint, filepath = save_as)
        
        plot(
            num_episodes = e,
            rewards = rewards_list,
            filepath = save_as
            )

        self.env.close()
        return rewards_list

    def play(self, filepath, num_episodes=1):
        with open(filepath, 'rb') as inp:
            checkpoint = pickle.load(inp)
        
        q_table = checkpoint["q_table"]
        for _ in range(num_episodes):
            t = 0
            done = False
            state = self.env.reset()
            while not done:
                self.env.render()
                t += 1
                action = np.argmax(q_table.get(state, 0))
                state, _, done, _ = self.env.step(action)
            print("Done at step {}".format(t))
        self.env.close()
        
    def save(self, checkpoint, filepath):
        with open("models/" + filepath + ".pkl", 'wb') as outp:
            pickle.dump(checkpoint, outp, pickle.HIGHEST_PROTOCOL)


class Vanilla_DQN_Agent:
    def __init__(
        self, environment, model_class, learning_rate, epsilon_schedule,
        discount_factor=0.9, replay_buffer_size=50000, minimum_buffer_size=5000,
        batch_size=128, update_frequency=10, device = torch.device('cpu'), seed=None
        ):
        self.env = environment
        self.state_shape = self.env.observation_space.shape
        self.minimum_buffer_size = minimum_buffer_size
        self.discount_factor = discount_factor
        self.epsilon_schedule = epsilon_schedule
        self.device = device
        self.online_model = model_class().to(self.device)
        self.optimizer = torch.optim.Adam(self.online_model.parameters(), lr = learning_rate)
        self.target_model = model_class().to(self.device)
        self.target_model.load_state_dict(self.online_model.state_dict())
        self.target_model.eval()
        self.update_frequency = update_frequency
        self.seed(seed)
        self.memory = ExperienceReplayBuffer(replay_buffer_size, self.state_shape, batch_size, self.rng)
    
    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed) if seed else np.random.default_rng()
    
    def get_action(self, state, epsilon):
        gen = self.rng.uniform(0, 1)
        if gen < epsilon:
            return self.env.action_space.sample()

        inp = torch.from_numpy(np.expand_dims(state, axis=0))
        inp = inp.to(self.device).float()
        with torch.no_grad():
            q_values = self.online_model(inp)
        
        return torch.argmax(q_values).item()
    
    def get_online_q_value(self, states, actions):
        inp = states.to(self.device).float()
        q_values = self.online_model(inp)
        return q_values[range(len(states)), actions]
    
    def get_target_q_value(self, states):
        inp = states.to(self.device).float()
        with torch.no_grad():
            q_values = self.target_model(inp)
            
        return torch.max(q_values, axis = 1).values

    def update_target_model(self):
        self.target_model.load_state_dict(self.online_model.state_dict())
    
    def train(self, num_episodes, save_as):
        rewards_list = []
        steps = 0

        for e in tqdm.tqdm(range(1, num_episodes + 1)):
            state = self.env.reset()
            done = False
            episode_reward = 0
            epsilon = self.epsilon_schedule(e)
            while not done:
                steps += 1
                action = self.get_action(state, epsilon)
                if SHOW_EVERY and e % SHOW_EVERY == 0:
                    self.env.render()
                new_state, reward, done, _ = self.env.step(action)
                self.memory.push(state, action, reward, new_state, done)
                episode_reward += reward
                state = new_state

                if len(self.memory) >= self.minimum_buffer_size and steps % self.update_frequency == 0:
                    states, actions, rewards, new_states, mask = self.memory.sample()
                    rewards = rewards.to(self.device)
                    mask = mask.to(self.device)
                    target_q_values = self.get_target_q_value(new_states) * mask
                    q_values = self.get_online_q_value(states, actions)
                    loss = (
                        rewards + self.discount_factor * target_q_values - q_values
                    ).square().mean()
                    self.optimizer.zero_grad()
                    loss.backward()

                    # clip gradients
                    torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), 1)

                    self.optimizer.step()

                    # update weights of the target model
                    self.update_target_model()
            
            rewards_list.append(episode_reward)
            if e % 100 == 0:
                checkpoint = {
                    "model_state": self.online_model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "hyperparameters": {
                        "batch_size": self.memory.batch_size,
                        "buffer_size": self.memory.capacity,
                        "discount_factor": self.discount_factor,
                        "update_frequency": self.update_frequency
                        }
                    }
                self.save(checkpoint = checkpoint, filepath = save_as + "/" + str(e))

                plot(
                    num_episodes = e,
                    rewards = rewards_list,
                    filepath = save_as
                    )

        self.env.close()
        return rewards_list

    def play(self, model_class, filepath, num_episodes=1):
        model = model_class()
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        for _ in range(num_episodes):
            t = 0
            done = False
            state = self.env.reset()
            while not done:
                self.env.render()
                t += 1
                state = np.array(state)
                inp = torch.from_numpy(np.expand_dims(state, axis=0)).float()
                action = torch.argmax(model(inp)).item()
                state, _, done, _ = self.env.step(action)
            print("Done at step {}".format(t))
        self.env.close()
        
    def save(self, checkpoint, filepath):
        torch.save(checkpoint, "models/" + filepath + ".pth")


class Double_DQN_Agent(Vanilla_DQN_Agent):    
    def get_best_actions(self, states):
        inp = states.to(self.device).float()
        with torch.no_grad():
            q_values = self.online_model(inp)
            
        return torch.argmax(q_values, axis = 1)
    
    def get_target_q_value(self, states):
        inp = states.to(self.device).float()
        actions = self.get_best_actions(inp)
        with torch.no_grad():
            q_values = self.target_model.forward(inp)

        return q_values[range(len(states)), actions]


class Double_DQN_Priority_Agent(Double_DQN_Agent):
    def __init__(
        self, environment, model_class, learning_rate, epsilon_schedule, beta_schedule,
        discount_factor=0.9, replay_buffer_size=50000, minimum_buffer_size=5000,
        batch_size=128, alpha=0.0, update_frequency=10, device = torch.device('cpu'), seed=None
        ):
        self.env = environment
        self.state_shape = self.env.observation_space.shape
        self.minimum_buffer_size = minimum_buffer_size
        self.discount_factor = discount_factor
        self.epsilon_schedule = epsilon_schedule
        self.device = device
        self.online_model = model_class().to(self.device)
        self.optimizer = torch.optim.Adam(self.online_model.parameters(), lr = learning_rate)
        self.target_model = model_class().to(self.device)
        self.target_model.load_state_dict(self.online_model.state_dict())
        self.target_model.eval()
        self.update_frequency = update_frequency
        self.beta_schedule = beta_schedule
        self.seed(seed)
        self.memory = PrioritizedExperienceReplayBuffer(replay_buffer_size, self.state_shape, batch_size, alpha, self.rng)
    
    def train(self, num_episodes, save_as):
        rewards_list = []
        steps = 0

        for e in tqdm.tqdm(range(1, num_episodes + 1)):
            state = self.env.reset()
            done = False
            episode_reward = 0
            epsilon = self.epsilon_schedule(e)
            beta = self.beta_schedule(e)
            while not done:
                steps += 1
                action = self.get_action(state, epsilon)
                if SHOW_EVERY and e % SHOW_EVERY == 0:
                    self.env.render()
                new_state, reward, done, _ = self.env.step(action)
                self.memory.push(state, action, reward, new_state, done)
                episode_reward += reward
                state = new_state

                if len(self.memory) >= self.minimum_buffer_size and steps % self.update_frequency == 0:
                    states, actions, rewards, new_states, mask, indices, weights = self.memory.sample(beta = beta)
                    rewards = rewards.to(self.device)
                    mask = mask.to(self.device)
                    weights = weights.to(self.device)
                    target_q_values = self.get_target_q_value(new_states) * mask
                    q_values = self.get_online_q_value(states, actions)
                    deltas = rewards + self.discount_factor * target_q_values - q_values
                    priorities = deltas.abs().cpu().detach()
                    self.memory.update(indices = indices, priorities = priorities + 1e-6)
                    self.optimizer.zero_grad()
                    loss = (
                        deltas * weights
                    ).pow(2).mean()
                    loss.backward()

                    # clip gradients
                    torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), 1)

                    self.optimizer.step()

                    # update weights of the target model
                    self.update_target_model()
            
            rewards_list.append(episode_reward)
            if e % 100 == 0:
                checkpoint = {
                    "model_state": self.online_model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "hyperparameters": {
                        "batch_size": self.memory.batch_size,
                        "alpha": self.memory.alpha,
                        "buffer_size": self.memory.capacity,
                        "discount_factor": self.discount_factor,
                        "update_frequency": self.update_frequency
                        }
                    }
                self.save(checkpoint = checkpoint, filepath = save_as + "/" + str(e))

                plot(
                    num_episodes = e,
                    rewards = rewards_list,
                    filepath = save_as
                    )

        self.env.close()
        return rewards_list