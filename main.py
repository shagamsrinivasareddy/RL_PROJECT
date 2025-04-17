import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from queue import PriorityQueue

# Pygame Setup
WIDTH, HEIGHT = 600, 600  # Adjust size to fit 6x6 grid
GRID_SIZE = 6  # Set grid size to 6x6
CELL_SIZE = WIDTH // GRID_SIZE
FPS = 15

# Environment Constants
EMPTY, OBSTACLE, DRONE, TARGET = 0, 1, 2, 3

# Colors (used if no image loaded)
WHITE = (255, 255, 255)

# Deep Q-Network Parameters
EPSILON = 0.1
GAMMA = 0.95
LR = 0.001

# Pygame Init
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Autonomous Drone Navigation")
clock = pygame.time.Clock()

# Load Images
drone_img = pygame.transform.scale(pygame.image.load(r"C:\Users\Admin\OneDrive\Documents\rl\assets\drone.png"), (CELL_SIZE, CELL_SIZE))
obstacle_img = pygame.transform.scale(pygame.image.load(r"C:\Users\Admin\OneDrive\Documents\rl\assets\obstacle.png"), (CELL_SIZE, CELL_SIZE))
target_img = pygame.transform.scale(pygame.image.load(r"C:\Users\Admin\OneDrive\Documents\rl\assets\target.png"), (CELL_SIZE, CELL_SIZE))
bg_img = pygame.transform.scale(pygame.image.load(r"C:\Users\Admin\OneDrive\Documents\rl\assets\path.jpeg"), (WIDTH, HEIGHT))

# DQN Model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Environment Setup
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

def place_randomly(value):
    while True:
        x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
        if grid[x, y] == EMPTY:
            grid[x, y] = value
            return x, y

# Initialize environment
drone_x, drone_y = place_randomly(DRONE)
target_x, target_y = place_randomly(TARGET)

# Place obstacles (randomly in the grid, but not in the drone or target position)
for _ in range(8):  # number of obstacles
    place_randomly(OBSTACLE)

# Agent & DQN
agent = DQN(4, 4)  # Input: drone_x, drone_y, target_x, target_y | Output: 4 directions
optimizer = optim.Adam(agent.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# Movement vectors
MOVES = [(0, -1), (0, 1), (-1, 0), (1, 0)]

def get_state():
    return torch.tensor([drone_x, drone_y, target_x, target_y], dtype=torch.float32)

def valid(x, y):
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and grid[x, y] != OBSTACLE

def a_star(start, goal):
    queue = PriorityQueue()
    queue.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while not queue.empty():
        _, current = queue.get()
        if current == goal:
            break

        for dx, dy in MOVES:
            next_node = (current[0] + dx, current[1] + dy)
            if valid(*next_node):
                new_cost = cost_so_far[current] + 1
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + abs(goal[0] - next_node[0]) + abs(goal[1] - next_node[1])
                    queue.put((priority, next_node))
                    came_from[next_node] = current

    path = []
    current = goal
    while current != start:
        if current not in came_from:
            return []
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

def step(action):
    global drone_x, drone_y
    dx, dy = MOVES[action]
    new_x, new_y = drone_x + dx, drone_y + dy
    reward = -1
    done = False

    if valid(new_x, new_y):
        grid[drone_x, drone_y] = EMPTY
        drone_x, drone_y = new_x, new_y
        if (drone_x, drone_y) == (target_x, target_y):
            reward = 100
            done = True
        grid[drone_x, drone_y] = DRONE
    else:
        reward = -5  # penalty for hitting wall/obstacle
    return reward, done

# Main Loop
running = True
max_episodes=10
episode = 0
while running and episode < max_episodes:
    screen.blit(bg_img, (0, 0)) # Draw background first

    # DQN Decision
    state = get_state()
    if random.random() < EPSILON:
        action = random.randint(0, 3)
    else:
        with torch.no_grad():
            action = torch.argmax(agent(state)).item()

    # Take a step based on the action
    reward, done = step(action)  # Ensure 'done' is updated here

    next_state = get_state()
    target_q = agent(state).clone()
    target_q[action] = reward + GAMMA * torch.max(agent(next_state)).item()
    loss = loss_fn(agent(state), target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # A* Visualization (path)
    path = a_star((drone_x, drone_y), (target_x, target_y))

    # Clear the grid before drawing
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            cell_type = grid[x, y]
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if cell_type == OBSTACLE:
                # Draw obstacle (make sure obstacles are drawn first)
                screen.blit(obstacle_img, rect)  # Draw obstacle image
            elif cell_type == TARGET:
                # Draw target as an image
                screen.blit(target_img, rect)  # Draw target image

    # Now draw the path (highlighted in cyan)
    for px, py in path:
        pygame.draw.rect(screen, (0, 255, 255), (px * CELL_SIZE, py * CELL_SIZE, CELL_SIZE, CELL_SIZE), 2)

    # Draw the drone image at its new position
    rect = pygame.Rect(drone_x * CELL_SIZE, drone_y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    screen.blit(drone_img, rect)

    # Refresh the screen with the new drawings
    pygame.display.flip()
    clock.tick(FPS)

    if done:
        print(f"🎯 Target reached in episode {episode + 1}")
        episode += 1
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        drone_x, drone_y = place_randomly(DRONE)
        target_x, target_y = place_randomly(TARGET)
        for _ in range(8):  # obstacles
            place_randomly(OBSTACLE)

    # Optionally show message on last episode
        if episode >= max_episodes:
            print("✅ Finished all 10 episodes!")
            running = False

pygame.quit()
