#general
import numpy as np
from tqdm import tqdm

# dataset imports
import gym
import d4rl

from morel.morel import Morel
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F





class Maze2DDataset(Dataset):

    def __init__(self):
        self.env = gym.make('maze2d-umaze-v1')
        dataset = self.env.get_dataset()

        # # Input data
        # self.source_observation = dataset["observations"][:-1]
        # self.source_action = dataset["actions"][:-1]

        # # Output data
        # self.target_observation = dataset["observations"][1:]

    def __getitem__(self, idx):
        # feed = torch.FloatTensor(np.concatenate([self.source_observation[idx], self.source_action[idx]])).to("cuda:0")
        # target = torch.FloatTensor(self.target_observation[idx]).to("cuda:0" )
        # return feed, target
        return None, None

    def __len__(self):
        # return len(self.source_observation)
        return 5

# Instantiate dataset
dynamics_data = Maze2DDataset()
# print(dynamics_data.env.action_space)
dataloader = DataLoader(dynamics_data, batch_size=128, shuffle = True)

agent = Morel(4, 2)

agent.train(dataloader, dynamics_data.env)






