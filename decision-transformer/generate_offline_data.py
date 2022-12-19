import pickle
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from utils import flatten_state

num_games = 250
max_steps = 80


# for more info, please refer to github issue https://github.com/Farama-Foundation/Gymnasium/issues/77
env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

dataset = []


for game in range(num_games):
    row = {"observations": [], "actions": [], "rewards": [], "dones": []}
    env.reset()
    for step in range(max_steps):
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)

        observation = flatten_state(observation)

        row["observations"].append(observation)
        row["actions"].append([action])
        row["rewards"].append(reward)
        row["dones"].append(done)
        if done:
            break
    dataset.append(row)


with open('mario-v0-offline-random-dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

