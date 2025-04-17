Autonomous Drone Navigation using Deep Q-Networks (DQN)

This project demonstrates an autonomous drone navigating a 6x6 grid environment using a Deep Q-Network (DQN) for learning and decision-making. The drone learns to reach a target while avoiding randomly placed obstacles. Visualization is handled using Pygame with path planning shown via A* algorithm.

Features 6x6 grid environment

Random obstacle placement

Deep Q-Network-based learning agent

A* algorithm for optimal path visualization

Visual simulation using Pygame

Tech Stack Python

PyTorch (Deep Q-Learning)

Pygame (Visualization)

NumPy (Grid management)

Requirements Make sure you have the following installed: pip install pygame torch numpy

Assets Place the following image assets in assets/ folder:

drone.png

obstacle.png

target.png

path.jpeg
How to Run
python drone_navigation.py


RL Output Video
[Watch the RL output video](assets/rl_output.mp4)

