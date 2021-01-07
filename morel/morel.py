# morel imports
from morel.models.Dynamics import DynamicsEnsemble
from morel.models.Policy import PPO2

# torch imports
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
class Morel():
    def __init__(self, obs_dim, action_dim, tensorboard_writer = None, comet_experiment = None):
        # self.dynamics = DynamicsEnsemble(obs_dim + action_dim, obs_dim)

        self.tensorboard_writer = tensorboard_writer
        self.comet_experiment = comet_experiment

        self.policy = PPO2(obs_dim, action_dim)

    def train(self, dataloader, env, log_to_tensorboard = False):
        writer = None
        if(log_to_tensorboard):
            writer = SummaryWriter()

        print("---------------- Beginning Dynamics Training ----------------")
        # self.dynamics.train(dataloader, epochs = 1, summary_writer = writer)

        print("---------------- Ending Dynamics Training ----------------")

        print("---------------- Beginning Policy Training ----------------")
        self.policy.train(env, summary_writer = self.tensorboard_writer, comet_experiment = self.comet_experiment)



