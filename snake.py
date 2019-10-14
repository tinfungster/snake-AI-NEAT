import pygame
import random
import math
import time
import neat
import pickle
import os
import numpy as np

class Snake:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.dir_x = 1
        self.dir_y = 0

        self.length = 1
        self.hunger = 100
        self.nodes = [(x,y)]
        
    # 0: forward, 1: left, 2: right    
    def move(self, action):
        angle = 0
        if action == 1:
            angle = -90
        if action == 2:
            angle = 90
        rad = math.radians(angle)

        new_x = self.dir_x * round(math.cos(rad)) - self.dir_y * round(math.sin(rad)); 
        new_y = self.dir_x * round(math.sin(rad)) + self.dir_y * round(math.cos(rad));
        self.dir_x = new_x
        self.dir_y = new_y
        
        self.x += self.dir_x
        self.y += self.dir_y
            
        if (self.x,self.y) in self.nodes:
            return False
            
        self.nodes.append((self.x, self.y))
        self.hunger -= 1
        return True
        
class Pill:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
class Game:
        
    def __init__(self, width, height, screen=None, player=None):
        self.width = width
        self.height = height
        self.player = player
        self.screen = screen
        self.board = [[]*self.height for x in range(self.width)]
        self.snake = None
        self.pill = None
        self.num_steps = 0
        self.fitness = 0
        self.apple_eaten = 0
        self.reset()
        
    def reset(self):
        self.apple_eaten = 0
        
        self.num_steps = 0
        self.fitness = 0
        self.snake = Snake(self.width//2, self.height//2)
        while True:
            x = random.randint(0, self.width-1)
            y = random.randint(0, self.height-1)
            if x != self.width//2 or y != self.height//2:
                break
        self.pill = Pill(x,y)
        if self.screen is not None:
            self.draw()
        
    def draw(self):
        # reset screen
        self.screen.fill(pygame.Color('black'))
        # draw snake
        for x,y in self.snake.nodes:
            pygame.draw.rect(self.screen, pygame.Color('white'), pygame.Rect(x*16,y*16,16,16))        
        # draw pill        
        pygame.draw.circle(self.screen, pygame.Color('white'), (self.pill.x*16+8,self.pill.y*16+8), 6)
        # update screen
        pygame.display.flip()
        
    
    def step(self,action=None):
        self.num_steps += 1
        if self.player is not None:
            action = self.player.get_action()
        valid = self.snake.move(action)
        self.check_pill()
        if (self.snake.x < 0 or self.snake.x >= self.width or
            self.snake.y < 0 or self.snake.y >= self.height or
            not valid or self.snake.hunger <= 0):
            return False   
        if self.screen is not None:
            self.draw()
        return True
        
    def check_pill(self):
        if self.snake.x == self.pill.x and self.snake.y == self.pill.y:
            valid = False
            while not valid:
                valid = True
                x = random.randint(0, self.width-1)
                y = random.randint(0, self.height-1)
                for s_x,s_y in self.snake.nodes:
                    if x == s_x and y == s_y:
                        valid = False
                        break
            self.pill = Pill(x,y)
            self.snake.length += 1
            self.snake.hunger += 100
            self.fitness += 1000
            self.apple_eaten += 1
        else:
            del self.snake.nodes[0]
            self.fitness += max(0,10 - round(((self.pill.x - self.snake.x)**2 + (self.pill.y - self.snake.y)**2)**.5))
            
    def get_normalized_state(self):

        # inputs = []

        # rad = math.radians(-90)
        # new_dir_x = self.snake.dir_x * round(math.cos(rad)) - self.snake.dir_y * round(math.sin(rad)); 
        # new_dir_y = self.snake.dir_x * round(math.sin(rad)) + self.snake.dir_y * round(math.cos(rad));
        # left_val = self.have_a_glance(new_dir_x, new_dir_y)
        # inputs += left_val

        # rad = math.radians(90)
        # new_dir_x = self.snake.dir_x * round(math.cos(rad)) - self.snake.dir_y * round(math.sin(rad)); 
        # new_dir_y = self.snake.dir_x * round(math.sin(rad)) + self.snake.dir_y * round(math.cos(rad));

        # right_val = self.have_a_glance(new_dir_x, new_dir_y)
        # inputs += right_val

        # forward_val = self.have_a_glance(self.snake.dir_x, self.snake.dir_y)
        # inputs += forward_val
        return self.vision_matrix()
    
    def have_a_glance(self, new_dir_x, new_dir_y):

        left_apple = 0
        left_tail = 0
        left_wall = 20
        new_x = self.snake.x
        new_y = self.snake.y

        while True:
            new_x = new_x + new_dir_x
            new_y = new_y + new_dir_y

            if new_x <= 0 or new_x >= self.width or new_y <= 0 or new_y >= self.height:
                left_wall = how_far(self.snake.x, self.snake.y, new_x, new_y)
                break
            
            if new_x == self.pill.x and new_y == self.pill.y:
                left_apple = 1
            
            if (new_x, new_y) in self.snake.nodes:
                left_tail = 1

        return [left_apple, left_tail, left_wall]


    def get_fitness(self):
        fitness = self.num_steps* + (2**self.apple_eaten+(self.apple_eaten**2.1)*500) - ((self.apple_eaten**1.2)*(0.25*self.num_steps)**1.3)
        return fitness

    def vision_matrix(self, distance = 5):

        obstacles = [[0 for i in range(distance)] for i in range(distance)]
        for r_id, row in enumerate(obstacles):
            for c_id, cell in enumerate(row):
                cell_x = (self.snake.x - 2) + c_id
                cell_y = (self.snake.y - 2) + r_id

                if cell_x == 0 or cell_x == self.width or cell_y == 0 or cell_y == self.height:
                    cell = 1
                
                if (cell_x, cell_y) in self.snake.nodes:
                    cell = 1

        flatten_obstacles = [value for row in obstacles for value in row]
        for row in obstacles:
            print(row)

        food = obstacles = [[0 for i in range(distance)] for i in range(distance)]
        for r_id, row in enumerate(obstacles):
            for c_id, cell in enumerate(row):
                cell_x = (self.snake.x - 2) + c_id
                cell_y = (self.snake.y - 2) + r_id

                if cell_x == self.pill.x and cell_y == self.pill.y:
                    cell = 1

        flatten_food = [value for row in food for value in row]
        
        return flatten_food + flatten_obstacles
    

        
class Player:

    def __init__(self):
        self.type = 0
        self.name = 'human'
    
    def get_action(self):
        pygame.event.clear()
        while True:
            event = pygame.event.wait()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    return 2
                elif event.key == pygame.K_a:
                    return 1
                elif event.key == pygame.K_w:
                    return 0


def how_far(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2+(y2-y1)**2)

def main():

    width = 25
    height = 25
    
    pygame.init()
    screen = pygame.display.set_mode((width * 16,height * 16))
    screen.fill(pygame.Color('black'))
    pygame.display.set_caption('Snake')
    pygame.display.flip()
    game = Game(width,height,screen)
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    with open('winner-feedforward', 'rb') as f:
        winner = pickle.load(f)
    
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    running = True
    while running:
        pygame.event.pump()
        print(f"SNAKE COORDS: {(game.snake.x, game.snake.y)}")
        print(f"")
        inputs = game.get_normalized_state()
        action = net.activate(inputs)
        running = game.step(np.argmax(action))

    
if __name__ == '__main__':
    main()