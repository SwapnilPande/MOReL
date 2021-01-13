import numpy as np

# torch imports
import torch


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
                        start_states,
                        timeout_steps = 300,
                        uncertain_penalty = -100,
                        device = "cuda:0"):
        self.dynamics_model = dynamics_model

        self.uncertain_penalty = uncertain_penalty
        self.start_states = start_states

        self.input_dim = self.dynamics_model.input_dim
        self.output_dim = self.dynamics_model.output_dim

        self.device = device

        # Save data transform parameters
        self.obs_mean = torch.Tensor(obs_mean).float().to(self.device)
        self.obs_std = torch.Tensor(obs_std).float().to(self.device)
        self.action_mean = torch.Tensor(action_mean).float().to(self.device)
        self.action_std = torch.Tensor(action_std).float().to(self.device)
        self.delta_mean = torch.Tensor(delta_mean).float().to(self.device)
        self.delta_std = torch.Tensor(delta_std).float().to(self.device)
        self.reward_mean = torch.Tensor([reward_mean]).float().to(self.device)
        self.reward_std = torch.Tensor([reward_std]).float().to(self.device)

        self.initial_obs_mean = torch.Tensor(initial_obs_mean).float().to(self.device)
        self.initial_obs_std = torch.Tensor(initial_obs_std).float().to(self.device)

        self.timeout_steps = timeout_steps

        self.state = None

    def reset(self):
        idx = np.random.choice(self.start_states.shape[0])
        next_obs = torch.tensor(self.start_states[idx]).float().to(self.device)
        self.state = (next_obs - self.obs_mean)/self.obs_std
        # print("reset!")
        # self.state = torch.normal(self.obs_mean, self.obs_std)
        # next_obs = self.obs_mean + self.obs_std*self.state
        self.steps_elapsed = 0

        return next_obs

    def step(self, action_unnormalized, obs = None):
        action = (action_unnormalized - self.action_mean)/self.action_std

        if obs is not None:
            self.state = (torch.tensor(obs).float().to(self.device) - self.obs_mean)/self.obs_std
            # self.state = torch.unsqueeze(self.state,0)
            print(self.state.shape)
            print(action.shape)

        predictions = self.dynamics_model.predict(torch.cat([self.state, action],0))


        deltas = predictions[:,0:self.output_dim-1]

        rewards = predictions[:,-1]

        # Calculate next state
        deltas_unnormalized = self.delta_std*torch.mean(deltas,0) + self.delta_mean
        state_unnormalized = self.obs_std*self.state + self.obs_mean
        next_obs = deltas_unnormalized + state_unnormalized
        self.state = (next_obs - self.obs_mean)/self.obs_std

        uncertain = self.dynamics_model.usad(predictions.cpu().numpy())

        reward_out = self.reward_std*torch.mean(rewards) + self.reward_mean

        if(uncertain):
            reward_out[0] = self.uncertain_penalty
        reward_out = torch.squeeze(reward_out)

        self.steps_elapsed += 1
        # print("reward {}\tuncertain{}".format(reward_out, uncertain))
        # input()

        return next_obs, reward_out, (uncertain or self.steps_elapsed > self.timeout_steps), {"HALT" : uncertain}