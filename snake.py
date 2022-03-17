from abc import abstractmethod
import numpy as np
import pygame

from gym import core, spaces
from pygame import gfxdraw

from utils import euclidean_distance
from collections import namedtuple


Position = namedtuple('Position', 'x, y')


class Snake(core.Env):

    RENDER_BLOCK_SIZE = 20
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, num_columns, num_rows, low, high, state_shape, seed=None):
        super().__init__()
        self.num_columns = num_columns
        self.num_rows = num_rows
        self.screen_width = self.num_columns * self.RENDER_BLOCK_SIZE
        self.screen_height = self.num_rows * self.RENDER_BLOCK_SIZE
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low = low, high = high, shape = state_shape, dtype = np.float64)
        self.max_distance = euclidean_distance((0, 0), (self.num_columns, self.num_rows))
        self.seed(seed)
        self.reset()
    
    @property
    @abstractmethod
    def state(self):
        raise NotImplementedError
    
    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed) if seed else np.random.default_rng()
    
    def reset(self):
        self.direction = 1
        self.screen = None
        self.clock = None
        self.isopen = True
        self.head = Position(self.num_columns // 2, self.num_rows // 2)
        self.body = [
            self.head,
            Position(self.head.x - 1, self.head.y),
            Position(self.head.x - 2, self.head.y)
            ]
        self.apple = None
        self._place_apple()
        self.distance = self.distance_to_apple()
        return self.state

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
        
    def _place_apple(self):
        self.apple = Position(self.rng.integers(low = 0, high = self.num_columns), self.rng.integers(low = 0, high = self.num_rows))
        while self.apple in self.body:
            self.apple = Position(self.rng.integers(low = 0, high = self.num_columns), self.rng.integers(low = 0, high = self.num_rows))
    
    def distance_to_apple(self):
        return euclidean_distance((self.head.x, self.head.y), (self.apple.x, self.apple.y)) / self.max_distance
    
    @property
    def length(self):
        return len(self.body)
    
    def render(self, mode = 'human'):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((0, 0, 0))
            
        l, r, t, b = self.apple.x * self.RENDER_BLOCK_SIZE, (self.apple.x + 1) * self.RENDER_BLOCK_SIZE, self.apple.y * self.RENDER_BLOCK_SIZE, (self.apple.y + 1) * self.RENDER_BLOCK_SIZE
        coords = [(l, b), (l, t), (r, t), (r, b)]
        gfxdraw.aapolygon(self.surf, coords, (255, 0, 0))
        gfxdraw.filled_polygon(self.surf, coords, (255, 0, 0))
        
        for i, j in self.body:
            l, r, t, b = i * self.RENDER_BLOCK_SIZE, (i + 1) * self.RENDER_BLOCK_SIZE, j * self.RENDER_BLOCK_SIZE, (j + 1) * self.RENDER_BLOCK_SIZE
            coords = [(l, b), (l, t), (r, t), (r, b)]
            gfxdraw.aapolygon(self.surf, coords, (0, 255, 0))
            gfxdraw.filled_polygon(self.surf, coords, (0, 255, 0))
            
        l, r, t, b = self.head.x * self.RENDER_BLOCK_SIZE, (self.head.x + 1) * self.RENDER_BLOCK_SIZE, self.head.y * self.RENDER_BLOCK_SIZE, (self.head.y + 1) * self.RENDER_BLOCK_SIZE
        coords = [(l, b), (l, t), (r, t), (r, b)]
        gfxdraw.aapolygon(self.surf, coords, (255, 255, 255))
        gfxdraw.filled_polygon(self.surf, coords, (255, 255, 255))

        # self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen
    
    def _crash(self):
        if self.head.x > self.num_columns - 1 or self.head.x < 0 or\
            self.head.y > self.num_rows - 1 or self.head.y < 0 or\
            self.head in self.body[1:]:
            return np.zeros(self.observation_space.shape), True
        
        return None, False
    
    def step(self, action):
        if action == 0 and self.direction == 1 or\
         action == 1 and self.direction == 0 or\
         action == 2 and self.direction == 3 or\
         action == 3 and self.direction == 2:
            action = self.direction
        
        self.direction = action
        
        x = self.head.x
        y = self.head.y
        if self.direction == 0:
            x -= 1
        elif self.direction == 1:
            x += 1
        elif self.direction == 2:
            y -= 1
        elif self.direction == 3:
            y += 1
            
        self.head = Position(x, y)
        self.body.insert(0, self.head)
        
        new_state, crashed = self._crash()
        if crashed:
            return new_state, -10, True, {}
        
        reward = 0
        if self.head == self.apple:
            reward = 10
            self._place_apple()
        else:
            self.body.pop()
            new_distance = self.distance_to_apple()
            reward = self.distance - new_distance
            self.distance = new_distance

        return self.state, reward, False, {}