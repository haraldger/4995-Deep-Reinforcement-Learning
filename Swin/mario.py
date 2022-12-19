import sys
import random
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import torch
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from swin_agent import SwinAgent
from epsilon_scheduler import EpsilonScheduler
from experience_replay import ReplayBuffer

# Swin Transformer
sys.path.append('./Swin-Transformer')
from models.swin_transformer_v2 import SwinTransformerV2 as Transformer

# Variables
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GAMES = 1000
REPLAY_MEMORY = 40000
INITIAL_EXPLORATION = 0
# INITIAL_EPSILON = 1.0
# FINAL_EPSILON = 0.01
DECAY_FRAMES = 10000
DECAY_MODE = 'multiple'
DECAY_RATE = 0.25
DECAY_START_FRAMES = 0
SYNC_FREQUENCY = 2500

# Data collection
reward_data = np.array([[0,0]])

def main():
    mario()

def mario():
    # Init data structures
    q_network = get_model()
    target_network = get_model()
    epsilon_scheduler = EpsilonScheduler(decay_frames=DECAY_FRAMES, decay_mode=DECAY_MODE, decay_rate=DECAY_RATE, start_frames=DECAY_START_FRAMES)
    replay_buffer = ReplayBuffer(capacity=REPLAY_MEMORY)
    agent = SwinAgent(q_network, target_network, epsilon_scheduler, replay_buffer, num_actions=7,
                        initial_exploration=INITIAL_EXPLORATION, sync_frequency=SYNC_FREQUENCY)

    # Environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    next_state = env.reset()
    next_state = process_state(next_state)

    total_reward = 0
    for game in range(GAMES):
        if game % 10 == 0:
            print(f'Game {game}')

        for frame in range(3000):   # Max frames per game

            previous_state = next_state

            # Act
            action = agent.act(previous_state)
            next_state, reward, terminated, info = env.step(action)
            next_state = process_state(next_state)

            total_reward += reward

            # Experience replay
            replay_buffer.add(previous_state, action, next_state, reward)

            # Learn
            agent.learn()

            # Update frame counters
            agent.step()
            epsilon_scheduler.step()

            if terminated:
                print("Terminated naturally")
                break 


        # Data collection
        global reward_data
        reward_data = np.concatenate((reward_data, np.array([[game, total_reward]])))
        plt.figure()
        plt.plot(reward_data[:,0], reward_data[:,1])
        plt.savefig(f'data/mario_graph.png')
        plt.close()

        total_reward = 0
        next_state = env.reset() 
        next_state = process_state(next_state)
        
    
def process_state(state):
    state = resize(state, (84, 84))
    state = np.moveaxis(state, 2, 0)
    return state


def get_model(image_size=(84,84), patch_size=3, in_channels=3,
            num_actions=8, depths=[2,3,2], heads=[3,3,6],
            window_size=7, mlp_ratio=4, drop_path_rate=0.1):
    """
    Default settings are appropriate for Atari games. For other environments, change patch size
    and window size to be compatible, as well as image size to match the environment. Pre-
    processing of the environment may be necessary, preferred format is 256x256, or 84x84 scaled
    by a factor of two (168x168, 336x336, etc.).
    num_actions corresponds to actions available in the environment.
    in_channels should be 1, 3 or 4, for greyscale, RGB, etc.
    """
    return Transformer(img_size=image_size, patch_size=patch_size, in_chans=in_channels,
                    num_classes=num_actions, depths=depths, num_heads=heads, window_size=window_size,
                    mlp_ratio=mlp_ratio, drop_path_rate=drop_path_rate).to(DEVICE)




if __name__ == "__main__":
    main()