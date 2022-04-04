import random
import pygame

from pygame import gfxdraw
from enum import Enum
from collections import namedtuple


Position = namedtuple('Position', 'x, y')


class Snake:

    RENDER_BLOCK_SIZE = 20
    RENDER_OUTLINE = 1

    def __init__(self, num_columns, num_rows):
        super().__init__()
        self.num_columns = num_columns
        self.num_rows = num_rows
        self.screen_width, self.screen_height = self.num_columns * self.RENDER_BLOCK_SIZE, self.num_rows * self.RENDER_BLOCK_SIZE
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        self.direction = 1
        self.head = Position(self.num_columns // 2, self.num_rows // 2)
        self.body = [
            self.head,
            Position(self.head.x - 1, self.head.y),
            Position(self.head.x - 2, self.head.y)
            ]
        self.apple = None
        self._place_apple()
        
    def _place_apple(self):
        self.apple = Position(random.randint(0, self.num_columns - 1), random.randint(0, self.num_rows - 1))
        while self.apple in self.body:
            self.apple = Position(random.randint(0, self.num_columns - 1), random.randint(0, self.num_rows - 1))
        
    def step(self):
        action = self.direction
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = 0
                elif event.key == pygame.K_RIGHT:
                    action = 1
                elif event.key == pygame.K_UP:
                    action = 2
                elif event.key == pygame.K_DOWN:
                    action = 3
                    
        if action == 0 and self.direction == 1 or\
         action == 1 and self.direction == 0 or\
         action == 2 and self.direction == 3 or\
         action == 3 and self.direction == 2:
            action = self.direction
        
        self.direction = action
        
        x = self.head.x
        y = self.head.y
        if action == 1:
            x += 1
        elif action == 0:
            x -= 1
        elif action == 3:
            y += 1
        elif action == 2:
            y -= 1
            
        self.head = Position(x, y)
        self.body.insert(0, self.head)
        
        crashed = self._crash()
        if crashed:
            print(self.length - 3)
            return True
            
        if self.head == self.apple:
            self._place_apple()
        else:
            self.body.pop()
        
        self.render()

        return False
    
    @property
    def length(self):
        return len(self.body)
    
    def render(self):
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((0, 0, 0))
            
        l, r, t, b = self.apple.x * self.RENDER_BLOCK_SIZE + self.RENDER_OUTLINE, (self.apple.x + 1) * self.RENDER_BLOCK_SIZE - self.RENDER_OUTLINE, self.apple.y * self.RENDER_BLOCK_SIZE + self.RENDER_OUTLINE, (self.apple.y + 1) * self.RENDER_BLOCK_SIZE - self.RENDER_OUTLINE
        coords = [(l, b), (l, t), (r, t), (r, b)]
        gfxdraw.aapolygon(self.surf, coords, (255, 0, 0))
        gfxdraw.filled_polygon(self.surf, coords, (255, 0, 0))
        
        for i, j in self.body:
            l, r, t, b = i * self.RENDER_BLOCK_SIZE + self.RENDER_OUTLINE, (i + 1) * self.RENDER_BLOCK_SIZE - self.RENDER_OUTLINE, j * self.RENDER_BLOCK_SIZE + self.RENDER_OUTLINE, (j + 1) * self.RENDER_BLOCK_SIZE - self.RENDER_OUTLINE
            coords = [(l, b), (l, t), (r, t), (r, b)]
            gfxdraw.aapolygon(self.surf, coords, (0, 255, 0))
            gfxdraw.filled_polygon(self.surf, coords, (0, 255, 0))
            
        l, r, t, b = self.head.x * self.RENDER_BLOCK_SIZE + self.RENDER_OUTLINE, (self.head.x + 1) * self.RENDER_BLOCK_SIZE - self.RENDER_OUTLINE, self.head.y * self.RENDER_BLOCK_SIZE + self.RENDER_OUTLINE, (self.head.y + 1) * self.RENDER_BLOCK_SIZE - self.RENDER_OUTLINE
        coords = [(l, b), (l, t), (r, t), (r, b)]
        gfxdraw.aapolygon(self.surf, coords, (255, 255, 255))
        gfxdraw.filled_polygon(self.surf, coords, (255, 255, 255))

        # self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        pygame.display.flip()
        self.clock.tick(10)
    
    def _crash(self):
        if self.head.x > self.num_columns - 1 or self.head.x < 0 or\
            self.head.y > self.num_rows - 1 or self.head.y < 0 or\
            self.head in self.body[1:]:
            return True
        
        return False

if __name__ == "__main__":
    pygame.init()

    game = Snake(20, 20)
    done = False
    while not done:
        done = game.step()
        
    pygame.quit()