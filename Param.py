import gym
import torch
from torchvision import transforms as transf
import numpy as np

BATCH_SIZE = 48
LEARNING_RATE = 1e-3
GAMMA = 0.5
EPS = 0.9
EPS_DECAY = 0.999
MEMORY_SIZE = 1024
UPDATE = 10
IMG_SIZE = [72, 72]
NUM_EPISODE = 300
DEBUG = False

def get_env(name = 'CartPole-v0'):
    env = gym.make(name)
    env.reset()
    return env

if torch.cuda.is_available() and not DEBUG:
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

def get_screen(env):
    screen = env.render(mode='rgb_array')
    screen = screen.transpose((2, 0, 1)) # (HWC)->(CHW)
    screen = torch.from_numpy(screen.astype(np.float32) / 255)
    screen = transf.ToPILImage()(screen)
    screen = transf.Resize(IMG_SIZE)(screen)
    screen = transf.ToTensor()(screen).view(1, 3, IMG_SIZE[0], IMG_SIZE[1])
    return screen.to(device)
