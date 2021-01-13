# morel imports
from numpy.lib.npyio import save
from morel.models.Dynamics import DynamicsEnsemble
from morel.models.Policy import PPO2
from morel.fake_env import FakeEnv

import numpy as np
from tqdm import tqdm
import os

# torch imports
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

class Morel():
    def __init__(self, obs_dim, action_dim, tensorboard_writer = None, comet_experiment = None):
        self.tensorboard_writer = tensorboard_writer
        self.comet_experiment = comet_experiment

        self.dynamics = DynamicsEnsemble(obs_dim + action_dim, obs_dim+1, threshold = 1.0)
        self.policy = PPO2(obs_dim, action_dim)

    def train(self, dataloader, dynamics_data, log_to_tensorboard = False):
        if(self.comet_experiment is not None):
            self.comet_experiment.log_parameter("uncertain_penalty", -50)

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
                            self.dynamics_data.initial_obs_std,
                            self.dynamics_data.source_observation,
                            uncertain_penalty=-50.0)

        print("---------------- Beginning Policy Training ----------------")
        self.policy.train(env, summary_writer = self.tensorboard_writer, comet_experiment = self.comet_experiment)
        print("---------------- Ending Policy Training ----------------")

        print("---------------- Successfully Completed Training ----------------")

    def eval(self, env):#dynamics_data, compare_model= False):
        # self.dynamics_data = dynamics_data
        # real_env = dynamics_data.env
        # if(compare_model):
        #     fake_env = FakeEnv(self.dynamics,
        #                     self.dynamics_data.observation_mean,
        #                     self.dynamics_data.observation_std,
        #                     self.dynamics_data.action_mean,
        #                     self.dynamics_data.action_std,
        #                     self.dynamics_data.delta_mean,
        #                     self.dynamics_data.delta_std,
        #                     self.dynamics_data.reward_mean,
        #                     self.dynamics_data.reward_std,
        #                     self.dynamics_data.initial_obs_mean,
        #                     self.dynamics_data.initial_obs_std,
        #                     self.dynamics_data.source_observation,
        #                     uncertain_penalty=-50.0)

        # for i in range(50):
        #     real_obs = real_env.reset()
        #     fake_env.reset()
        #     done = False

        #     while(not done):
        #         input_obs = real_obs
        #         action = self.policy.eval(input_obs)
        #         if(compare_model):
        #             fake_obs, fake_reward, _, info  = fake_env.step(action, obs = real_obs)

        #         real_obs, real_reward, done, _ = real_env.step(action.cpu().numpy())
        #         real_env.render()

        #         if compare_model:
        #             print("Obs: {} {}".format(real_obs, fake_obs))
        #             print("Reward: {} {}".format(real_reward, fake_reward))
        #             print("USAD: {}".format(info["HALT"]))
        #             input()



        print("---------------- Beginning Policy Evaluation ----------------")
        total_rewards = []
        for i in tqdm(range(50)):
            _, _, _, _, _, _, _, info = self.policy.generate_experience(env, 1024, 0.95, 0.99)
            total_rewards.extend(info["episode_rewards"])

            if(self.tensorboard_writer is not None):
                self.tensorboard_writer.add_scalar('Metrics/eval_episode_reward', sum(info["episode_rewards"])/len(info["episode_rewards"]), step = i)

            if(self.comet_experiment is not None):
                self.comet_experiment.log_metric('eval_episode_reward', sum(info["episode_rewards"])/len(info["episode_rewards"]), step = i)


        print("Final evaluation reward: {}".format(sum(total_rewards)/len(total_rewards)))

        print("---------------- Ending Policy Evaluation ----------------")

    def save(self, save_dir):
        if(not os.path.isdir(save_dir)):
            os.mkdir(save_dir)

        self.policy.save(save_dir)
        self.dynamics.save(save_dir)

    def load(self, load_dir):
        self.policy.load(load_dir)
        self.dynamics.load(load_dir)



