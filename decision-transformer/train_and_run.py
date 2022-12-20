import pickle
from datasets import load_dataset
from transformers import DecisionTransformerConfig, Trainer, TrainingArguments

from DecisionTransformerGymDataCollator import DecisionTransformerGymDataCollator
from TrainableDT import TrainableDT
from DLNNDataLoader import DLNNDataLoader



dataset = pickle.load(open("mario-v0-offline-random-dataset.pkl", "rb"))


collator = DecisionTransformerGymDataCollator(dataset)

config = DecisionTransformerConfig(state_dim=collator.state_dim, act_dim=collator.act_dim)
model = TrainableDT(config)


training_args = TrainingArguments(
    output_dir="output/",
    remove_unused_columns=False,
    num_train_epochs=120,
    per_device_train_batch_size=64,
    learning_rate=1e-4,
    weight_decay=1e-4,
    warmup_ratio=0.1,
    optim="adamw_torch",
    max_grad_norm=0.25,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
)


print("Start Training")
trainer.train()
print("Done Training")






import numpy as np
from nes_py.wrappers import JoypadSpace
from utils import get_action, flatten_state
from PIL import Image
import torch

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


print("Start Playing")

# build the environment
model = model.to("cpu")
env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)


max_ep_len = 1000
device = "cpu"
scale = 1000.0 
TARGET_RETURN = 12000 / scale 

state_dim = collator.state_dim
act_dim = collator.act_dim

state_mean = collator.state_mean.astype(np.float32)
state_mean = torch.from_numpy(state_mean).to(device=device)

state_std = collator.state_std.astype(np.float32)
state_std = torch.from_numpy(state_std).to(device=device)


# Interact with the environment and create a video
episode_return, episode_length = 0, 0
state = flatten_state(env.reset()[0])

target_return = torch.tensor(TARGET_RETURN, device=device, dtype=torch.float32).reshape(1, 1)
states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
actions = torch.zeros((0, act_dim), device=device)
rewards = torch.zeros(0, device=device, dtype=torch.float32)


timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
for t in range(max_ep_len):
    actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
    rewards = torch.cat([rewards, torch.zeros(1, device=device)])

    action = get_action(
        model,
        (states - state_mean) / state_std,
        actions,
        rewards,
        target_return,
        timesteps,
    )
    actions[-1] = action.detach().cpu()
    action = torch.argmax(torch.nn.functional.softmax(action, dim=0)).item()


    action = env.action_space.sample()
    original_state, reward, done, _, _ = env.step(action)

    state = flatten_state(original_state)
    cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)

    # store every 100 frame
    if (t % 50 == 0) or (done) or (t == max_ep_len - 1):
        img = Image.fromarray(original_state)
        img.save(f'output/frame{t}.jpg')


    states = torch.cat([states, cur_state], dim=0)
    rewards[-1] = reward

    pred_return = target_return[0, -1] - (reward / scale)
    target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
    timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

    episode_return += reward
    episode_length += 1

    if done:
        break
