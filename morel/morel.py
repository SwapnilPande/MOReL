# morel imports
from morel.models.Dynamics import DynamicsEnsemble
from morel.models.Policy import PPO2

import numpy as np

# torch imports
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F


class FakeEnv:
    def __init__(self, dynamics_model,
                        obs_mean,
                        obs_std,
                        action_mean,
                        action_std,
                        delta_mean,
                        delta_std,
                        reward_mean,
                        reward_std,
                        initial_obs_mean,
                        initial_obs_std,
                        timeout_steps = 300):
        self.dynamics_model = dynamics_model

        self.input_dim = self.dynamics_model.input_dim
        self.output_dim = self.dynamics_model.output_dim

        # Save data transform parameters
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.action_mean = action_mean
        self.action_std = action_std
        self.delta_mean = delta_mean
        self.delta_std = delta_std
        self.reward_mean = reward_mean
        self.reward_std = reward_std

        self.initial_obs_mean = initial_obs_mean
        self.initial_obs_std = initial_obs_std

        self.timeout_steps = timeout_steps

        self.state = None

    def reset(self):
        self.state = np.random.normal(self.initial_obs_mean, self.initial_obs_std)
        self.steps_elapsed = 0

        return self.state

    def step(self, action):
        predictions = self.dynamics_model.predict(torch.FloatTensor(np.concatenate([self.state, action])).to("cuda:0"))

        deltas = predictions[:,0:self.output_dim-1]
        rewards = predictions[:,-1]

        # Calculate next state
        deltas_unnormalized = self.delta_std*deltas.mean(axis = 0) + self.delta_mean
        state_unnormalized = self.obs_std*self.state + self.obs_mean
        next_obs = deltas_unnormalized + state_unnormalized
        self.state = (next_obs - self.obs_mean)/self.obs_std

        uncertain = self.dynamics_model.usad(predictions)

        reward_out = self.reward_std*rewards.mean() + self.reward_mean

        if(uncertain):
            reward_out = -50.0

        self.steps += 1

        return self.state, reward_out, (uncertain or self.steps > self.timeout_steps), None

class Morel():
    def __init__(self, obs_dim, action_dim, tensorboard_writer = None, comet_experiment = None):
        self.tensorboard_writer = tensorboard_writer
        self.comet_experiment = comet_experiment

        self.dynamics = DynamicsEnsemble(obs_dim + action_dim, obs_dim+1)
        self.policy = PPO2(obs_dim, action_dim)

    def train(self, dataloader, dynamics_data, log_to_tensorboard = False):
        self.dynamics_data = dynamics_data

        print("---------------- Beginning Dynamics Training ----------------")
        self.dynamics.train(dataloader, epochs = 2, summary_writer = self.tensorboard_writer, comet_experiment = self.comet_experiment)
        print("---------------- Ending Dynamics Training ----------------")

        env = FakeEnv(self.dynamics,
                            self.dynamics_data.observation_mean,
                            self.dynamics_data.observation_std,
                            self.dynamics_data.action_mean,
                            self.dynamics_data.action_std,
                            self.dynamics_data.delta_mean,
                            self.dynamics_data.delta_std,
                            self.dynamics_data.reward_mean,
                            self.dynamics_data.reward_std,
                            self.dynamics_data.initial_obs_mean,
                            self.dynamics_data.initial_obs_std)

        print("---------------- Beginning Policy Training ----------------")
        self.policy.train(env, summary_writer = self.tensorboard_writer, comet_experiment = self.comet_experiment)



