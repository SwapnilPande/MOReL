from comet_ml import Experiment
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



experiment = Experiment(
    api_key="roUfs00hpDDbntO56E1FrQ29b",
    project_name="ppo2-debugging",
    workspace="swapnilpande",
)
experiment.set_name("var_exp_act")

class Morel():
    def __init__(self, obs_dim, action_dim):
        # self.dynamics = DynamicsEnsemble(obs_dim + action_dim, obs_dim)

        self.policy = PPO2(obs_dim, action_dim)

    def train(self, dataloader, env, log_to_tensorboard = False):
        writer = None
        if(log_to_tensorboard):
            writer = SummaryWriter()

        print("---------------- Beginning Dynamics Training ----------------")
        # self.dynamics.train(dataloader, epochs = 1, summary_writer = writer)

        print("---------------- Ending Dynamics Training ----------------")

        print("---------------- Beginning Policy Training ----------------")
        self.policy.train(env, experiment, summary_writer = writer)



