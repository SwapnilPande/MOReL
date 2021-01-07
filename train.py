
import argparse
import json
import subprocess
from comet_ml import Experiment

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

def main(args):
    # Create tensorboard writer if requested
    tensorboard_writer = None
    if(args.tensorboard):
        writer = SummaryWriter(log_dir = args.log_dir)

    # Create comet experiment if requested
    comet_experiment = None
    if(args.comet_config is not None):
        with open(args.comet_config, 'r') as f:
            comet_dict = json.load(f)
            experiment = Experiment(
                api_key = comet_dict["api_key"],
                project_name = comet_dict["project_name"],
                workspace = comet_dict["workspace"],
            )
            experiment.set_name(args.exp_name)

            # Get hash for latest git commit for logging
            last_commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8").rstrip()
            comet_experiment.log_parameter("git_commit_id", last_commit_hash)

    # Instantiate dataset
    dynamics_data = Maze2DDataset()

    dataloader = DataLoader(dynamics_data, batch_size=128, shuffle = True)

    agent = Morel(4, 2, tensorboard_writer = tensorboard_writer, comet_experiment = comet_experiment)

    agent.train(dataloader, dynamics_data.env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training')

    parser.add_argument('--log_dir', type=str, default='../results/')
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--comet_config', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default='exp_test')

    args = parser.parse_args()
    main(args)







