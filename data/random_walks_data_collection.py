import csv
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

num_games = 10000

for game in range(num_games):
    done = False 
    env.reset()
    total_reward = 0
    game_data = list()
    
    while not done:
        for step in range(5000):
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            total_reward += reward
            game_data.append([state, action, reward, done, info, 0])    # Index 5 is reward_to_go, calculated at the end

            if done:
                break
        done = True

    reward_to_go = 0
    for original_idx, row in reversed(list(enumerate(game_data))):
        row[5] = reward_to_go
        reward_to_go += row[2]

    with open('random_walks_data.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerows(game_data)

env.close()