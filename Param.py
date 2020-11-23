import gym
import torch
from torchvision import transforms as transf
import numpy as np

BATCH_SIZE = 32
LEARNING_RATE = 1e-2
GAMMA = 0.99
EPS = 0.1
MEMORY_SIZE = 4096
UPDATE = 16
IMG_SIZE = [120, 120]
NUM_EPISODE = 100

def get_env(name = 'CartPole-v0'):
    env = gym.make(name)
    # env = gym.wrappers.Monitor(env, directory=OUTDIR, force=True)
    env.reset()
    return env

if torch.cuda.is_available():
    device = torch.device("cuda")
    # device = torch.device("cpu")
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
