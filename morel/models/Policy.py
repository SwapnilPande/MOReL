import torch
import torch.nn as nn
import torch.nn.functional as F
from morel.fake_env import FakeEnv
# from torchviz import make_dot

import numpy as np
from tqdm import tqdm
import os

class ActorCriticPolicy(nn.Module):
    def __init__(self, input_dim,
                output_dim,
                n_neurons = 64,
                activation = nn.Tanh,
                distribution = torch.distributions.multivariate_normal.MultivariateNormal):
        # Validate inputs
        assert input_dim > 0
        assert output_dim > 0
        assert n_neurons > 0

        super(ActorCriticPolicy, self).__init__()

        # Store configuration parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_neurons = n_neurons
        self.distribution = distribution

        # Policy Network
        self.h0 = nn.Linear(input_dim, n_neurons)
        self.h0_act = activation()
        self.h1 = nn.Linear(n_neurons, n_neurons)
        self.h1_act = activation()
        self.output_layer = nn.Linear(n_neurons, output_dim)

        # Value Network
        self.h0v = nn.Linear(input_dim, n_neurons)
        self.h0_actv = activation()
        self.h1v = nn.Linear(n_neurons, n_neurons)
        self.h1_actv = activation()
        self.value_head = nn.Linear(n_neurons, 1)

        self.var = torch.nn.Parameter(torch.tensor([0.0,0.0]).cuda(), requires_grad = True)

        self.mean_activation = nn.Tanh()
        # self.var_activation = nn.Softplus()

    def forward(self, obs, action = None):
        # Policy Forward Pass
        x = self.h0(obs)
        x = self.h0_act(x)
        x = self.h1(x)
        x = self.h1_act(x)
        action_logit = self.output_layer(x)

        # Generate action distribution
        #TODO Add Support for additional distributions
        mean = action_logit[:,0:self.output_dim]
        var = torch.exp(self.var)
        action_dist = self.distribution(mean, torch.diag_embed(var))

        # Sample action if not passed as argument to function
        # Action is passed when doing policy updates
        if action is None:
            action = action_dist.sample()

        neg_log_prob = action_dist.log_prob(action) * -1.
        entropy = action_dist.entropy()

        # Value Forward Pass
        x = self.h0v(obs)
        x = self.h0_actv(x)
        x = self.h1v(x)
        x = self.h1_actv(x)
        value = self.value_head(x)
        value = torch.squeeze(value)

        return action, neg_log_prob, entropy, value

class ConvActorCriticPolicy(nn.Module):
    def __init__(self, input_dim,
                output_dim,
                n_neurons = 64,
                activation = nn.ReLU,
                distribution = torch.distributions.multivariate_normal.MultivariateNormal):
        # Validate inputs
        assert input_dim > 0
        assert output_dim > 0
        assert n_neurons > 0

        super(ConvActorCriticPolicy, self).__init__()

        # Store configuration parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_neurons = n_neurons
        self.distribution = distribution


        # Policy Network - cnn feature extractor followed by 2 fc layers
        self.cnn_p = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=3),
            nn.BatchNorm2d(16),
            activation(),
            nn.Conv2d(16, 32, kernel_size=5, stride=3),
            nn.BatchNorm2d(32),
            activation(),
            nn.Conv2d(32, 32, kernel_size=5, stride=3),
            nn.BatchNorm2d(32),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.fc1 = nn.Linear(32, n_neurons)
        self.fc1_act = activation()
        self.output_layer = nn.Linear(n_neurons, output_dim)

        # Value Network
        self.cnn_v = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=3),
            nn.BatchNorm2d(16),
            activation(),
            nn.Conv2d(16, 32, kernel_size=5, stride=3),
            nn.BatchNorm2d(32),
            activation(),
            nn.Conv2d(32, 32, kernel_size=5, stride=3),
            nn.BatchNorm2d(32),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.fc1_v = nn.Linear(32, n_neurons)
        self.fc1_act_v = activation()
        self.value_head = nn.Linear(n_neurons, 1)

        self.var = torch.nn.Parameter(torch.tensor([0.0]*self.output_dim), requires_grad = True)

        # self.mean_activation = nn.Tanh()
        # self.var_activation = nn.Softplus()

    def forward(self, obs, action = None):
        # Policy Forward Pass

        obs = obs.permute([0,3,1,2])
        x = self.cnn_p(obs)
        x = torch.squeeze(x,3)
        x = torch.squeeze(x,2)

        x = self.fc1(x)
        x = self.fc1_act(x)
        mean = self.output_layer(x)

        # Generate action distribution
        #TODO Add Support for additional distributions
        var = torch.exp(self.var)
        action_dist = self.distribution(mean, torch.diag_embed(var))

        # Sample action if not passed as argument to function
        # Action is passed when doing policy updates
        if action is None:
            action = action_dist.sample()

        neg_log_prob = action_dist.log_prob(action) * -1.
        entropy = action_dist.entropy()

        # Value Forward Pass
        x = self.cnn_p(obs)
        x = torch.squeeze(x,3)
        x = torch.squeeze(x,2)
        x = self.fc1_v(x)
        x = self.fc1_act_v(x)
        value = self.value_head(x)
        value = torch.squeeze(value)

        return action, neg_log_prob, entropy, value

class PPO2():
    def __init__(self, input_dim, output_dim, device = "cuda:0", network = ActorCriticPolicy):
        """Initializes a PPO2 Policy.

        Currently only has support for continuous action spaces.
        #TODO add support for discrete action spaces

        Args:
            input_dim: Dimension of observation
            output_dim: Dimension of action space
            device (str, optional): Device on which to allocate tensors. Defaults to "cuda:0".
        """
        # Store device on which to allocate tensors
        self.device = device

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Instantiate Actor Critic Policy
        self.policy = network(input_dim, output_dim, n_neurons=64).to(self.device)

    def forward(self, observation, action = None):
        """Performs a forward pass using the policy network

        Args:
            observation: torch tensor of dim (?,self.input_dim) containing observation.
            action (optional): Selected action of dim (?, self.output_dim) for which to calculate neg_log_prob of policy. If None, samples
            probability distribution to select action. Defaults to None.

        Returns:
            action: Selected action of dim (?, self.output_dim)
            neg_log_prob: Negative log probability of selected action (?,)
            entropy: Entropy of output distribution (?,)
            value: Estimated value of state (?,)
        """

        return self.policy(observation, action = action)

    def generate_experience(self, env, n_steps, gamma, lam):
        """Generates an experience rollout by querying the environment and passing actions generated by current policy.

        Args:
            env: Gym environment to generate experience from. Note this gym environment does need need to have the full API
                defined by a gym environment. It only uses the reset() and step functions. reset() only needs to return the
                next observation and step only needs to return the reward and next observation.
            n_steps: The number of steps to execute for rollout
            gamma: Discount factor used for generalized advantage estimate
            lam: Discount factor used for generalized advantage estimate

        Returns:
            mb_rewards: Reward collected at every time step
            mb_obs: Observation collected at every time step
            mb_returns: Q value estimate at every time step
            mb_done: Value of the "done" flag at every time step
            mb_actions: Action selected by policy at every time step
            mb_neg_log_prob: Negative log probability of selected action at every time step
            episode_reward: List of total episode rewards for every completed episode
        """

        # Reset environment on first step
        done = True

        # Initialize memory buffer
        mb_obs, mb_rewards, mb_actions, mb_values, mb_done, mb_neg_log_prob = [],[],[],[],[],[]
        info = {
            "episode_rewards" : [],
            "HALT" : 0
        }
        rewards = 0
        total_reward = 0
        env_info = {}

        # For n in range number of steps
        with torch.set_grad_enabled(False):
            for i in range(n_steps):
                if(done):

                    info["HALT"] += env_info.get("HALT", 0)

                    obs = env.reset()
                    done = False


                    # Convert obs to torch tensor
                    if(not isinstance(env, FakeEnv)):
                        obs = torch.from_numpy(obs.copy()).float().to(self.device)
                    obs = torch.unsqueeze(obs, 0)

                    info["episode_rewards"].append(total_reward)
                    total_reward = 0


                # Choose action
                action, neg_log_prob, _, value = self.forward(observation = obs)

                # # Retrieve values
                # action = np.squeeze(action.cpu().numpy())
                # neg_log_prob = np.squeeze(neg_log_prob.cpu().numpy())
                # value = np.squeeze(value.cpu().numpy())

                # # Append data from step to memory buffer
                # mb_obs.append(obs.copy())
                # mb_actions.append(action)
                # mb_values.append(value)
                # mb_neg_log_prob.append(neg_log_prob)
                # mb_done.append(done)

                # Retrieve values
                action = torch.squeeze(action)
                neg_log_prob = torch.squeeze(neg_log_prob)
                value = torch.squeeze(value)

                # Append data from step to memory buffer
                # mb_obs.append(obs.copy())
                mb_obs.append(obs)
                mb_actions.append(action)
                mb_values.append(value)
                mb_neg_log_prob.append(neg_log_prob)
                mb_done.append(done)

                # Step the environment, get new observation and reward
                # If we are interacting with a FakeEnv, we can safely keep the action as a torch tensor
                # Else, we must convert to a numpy array
                # If obs comes as numpy array, convert to torch tensor as well
                if(isinstance(env, FakeEnv)):
                    obs, rewards, done, env_info = env.step(action)

                    total_reward += rewards.cpu().item()
                else:
                    obs, rewards, done, env_info = env.step(action.cpu().numpy())
                    # from PIL import Image
                    # img = Image.fromarray(obs, 'RGB')
                    # img.show()
                    # input()
                    obs = torch.from_numpy(obs.copy()).float().to(self.device)

                    total_reward += rewards

                    rewards = torch.tensor(rewards).float().to(self.device)

                # Append reward to memory buffer as well
                mb_rewards.append(rewards)

                obs = torch.unsqueeze(obs, 0)
                if(self.render):
                    env.render()

            # # Convert memory buffer lists to numpy arrays
            # mb_obs = np.concatenate(mb_obs, 0).astype(np.float32)
            # mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
            # mb_actions = np.asarray(mb_actions)
            # mb_values = np.asarray(mb_values, dtype=np.float32)
            # mb_neg_log_prob = np.asarray(mb_neg_log_prob, dtype=np.float32)
            # mb_done = np.asarray(mb_done, dtype=np.bool)

            # Convert memory buffer lists to numpy arrays
            # print(mb_obs[0:5])
            mb_obs = torch.cat(mb_obs, 0)
            mb_rewards = torch.stack(mb_rewards)
            mb_actions = torch.stack(mb_actions)
            mb_values = torch.stack(mb_values)
            mb_neg_log_prob = torch.stack(mb_neg_log_prob)
            mb_done = np.asarray(mb_done, dtype=np.bool)

            # get value function for last state
            _, _, _, last_value = self.forward(obs)
            # last_value = last_value.cpu().numpy()

            # Compute generalized advantage estimate by bootstrapping
            mb_advs = torch.zeros_like(mb_rewards).float().to(self.device)
            last_gae_lam = torch.Tensor([0.0]).float().to(self.device)
            for t in reversed(range(n_steps)):
                # next_non_terminal stores index of the next time step (in reverse order) that is non-terminal
                if t == n_steps - 1:
                    # 1 if last step was non-terminal
                    # 0 if last step was terminal
                    next_non_terminal = 1.0 - done
                    next_values = last_value
                else:
                    next_non_terminal = 1.0 - mb_done[t+1]
                    next_values = mb_values[t+1]

                delta = mb_rewards[t] + gamma * next_values * next_non_terminal - mb_values[t]
                mb_advs[t] = last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam

            # compute value functions
            mb_returns = mb_advs + mb_values

        return mb_rewards, mb_obs, mb_returns, mb_done, mb_actions, mb_values, mb_neg_log_prob, info

    def train_step(self, clip_range,
                        entropy_coef,
                        value_coef,
                        obs,
                        returns,
                        dones,
                        old_actions,
                        old_values,
                        old_neg_log_probs):
        """Runs a single train step for the policy.

        Args:
            clip_range : Clip range for the value function and ratio in policy loss
            entropy_coef: Coefficient for entropy loss in total loss function
            value_coef: Coefficient for value loss in total loss function
            obs: Observations at each time step of rollout (n_steps,self.input_dim)
            returns: Estimated returns at each time step (n_steps, self.input_dim)
            dones: Done flag at each time step(n_steps, 1)
            old_actions: Action taken by agent at each time step (n_steps, self.output_dim)
            old_values: Value estimate of policy used to generate action at each time step (n_steps, self.output_dim)
            old_neg_log_probs: Negative log probabilities of actions for policy used to generate actions at each time step (n_steps, self.output_dim)

        Returns:
            loss: Total loss
            pg_loss: Policy loss
            value_loss: Value loss
            entropy_mean: Entropy loss
            approx_kl: Approximate KL Divergence between old and new policy
        """

        # Create torch tensors and send to correct device
        # returns = torch.tensor(returns).float().to(self.device)
        # old_actions = torch.tensor(old_actions).to(self.device)
        # old_values = torch.tensor(old_values).to(self.device)
        # old_neg_log_probs = torch.tensor(old_neg_log_probs).to(self.device)

        # Calculate and normalize the advantages
        with torch.set_grad_enabled(False):
            advantages = returns - old_values

            # Normalize the advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Set policy network to train mode
        self.policy.train()
        with torch.set_grad_enabled(True):
            # Feed batch through policy network
            actions, neg_log_probs, entropies, values = self.forward(obs, action = old_actions)

            loss, pg_loss, value_loss, entropy_mean, approx_kl = self.loss(clip_range,
                                                                            entropy_coef,
                                                                            value_coef,
                                                                            returns,
                                                                            values,
                                                                            neg_log_probs,
                                                                            entropies,
                                                                            advantages,
                                                                             old_values,
                                                                             old_neg_log_probs)

            # Backprop from loss
            loss.backward()

            return loss, pg_loss, value_loss, entropy_mean, approx_kl


    def train(self, env, optimizer = torch.optim.Adam,
                        lr =  0.00027,
                        n_steps = 1024,
                        time_steps = 1e6,
                        clip_range = 0.2,
                        entropy_coef = 0.01,
                        value_coef = 0.5,
                        num_batches = 4,
                        gamma = 0.99,
                        lam = 0.95,
                        max_grad_norm = 0.5,
                        num_train_epochs = 4,
                        comet_experiment = None,
                        summary_writer = None,
                        render = False):
        """Entry point for training the policy.

        The training routine is based off the the PPO2 implementation provided in the stable-baselines repo.
        The default hyperparameters are also set to match those of stable baselines.

        Args:
            env: Gym environment to generate experience from. Note this gym environment does need need to have the full API
                defined by a gym environment. It only uses the reset() and step functions. reset() only needs to return the
                next observation and step only needs to return the reward and next observation.
            optimizer (optional): Optimizer to optimize policy and value networks. Defaults to torch.optim.Adam.
            lr (float, optional): Learning rate. Defaults to 0.00025.
            n_steps (int, optional): Length of steps for rollouts from environment. Defaults to 1024.
            time_steps (int, optional): Total number of steps of experience to collect. Defaults to 1e6.
            clip_range (float, optional): Clip range for the value function and ratio in policy loss. Defaults to 0.2.
            entropy_coef (float, optional): Coefficient for entropy loss in total loss function. Defaults to 0.01.
            value_coef (float, optional): Coefficient for value loss in total loss function. Defaults to 0.5.
            num_batches (int, optional): Number of batches to split rollout into. Defaults to 4.
            gamma (float, optional): Discount factor used for generalized advantage estimate. Defaults to 0.99.
            lam (float, optional): Discount factor used for generalized advantage estimate. Defaults to 0.95.
            max_grad_norm (float, optional): Max gradient norm for gradient clipping. Defaults to 0.5.
            num_train_epochs (int, optional): Number of epochs to train for after each new experience rollout is generated. Defaults to 4.
            comet_experiment (optional) : A comet experiment object for logging. No logging if not passed. Defaults to None.
            summary_writer (optional): A tensorboard summary writer for logging. No logging if not passed. Defaults to None.
        """

        # Report hyperparameters for training
        if(comet_experiment is not None):

            hyper_params = {
                "learning_rate":  lr,
                "steps": time_steps,
                "n_steps (simulation)": n_steps,
                "clip_range" : clip_range,
                "max_grad_norm" : max_grad_norm,
                "num_batches" : num_batches,
                "gamma" : gamma,
                "lam" : lam,
                "num_train_epochs" : num_train_epochs,
                "value_coef" : value_coef,
                "entropy_coef" : entropy_coef
            }
            comet_experiment.log_parameters(hyper_params)

        self.render = render

        # Total number of train cycles to run
        n_updates = int(time_steps // n_steps)

        # Instantiate optimizer
        self.policy_optim = optimizer(self.policy.parameters(), lr = lr)

        # main train loop
        for update in tqdm(range(n_updates)):
            # Collect new experiences using the current policy
            rewards, obs, returns, dones, actions, values, neg_log_probs, info = self.generate_experience(env, n_steps, gamma, lam)
            indices = np.arange(n_steps)

            # Loop over train epochs
            for i in range(num_train_epochs):
                # Shuffle order of data
                np.random.shuffle(indices)

                # Calculate size of each batch
                batch_size = n_steps // num_batches

                # If not evenly divisible, add 1 sample to each batch, last batch will automatically be smaller
                if(n_steps % num_batches):
                    batch_size +=1

                # Loop over batches in single epoch
                for batch_num in range(num_batches):
                    # Reset gradients
                    self.policy.zero_grad()

                    # Get indices for batch
                    if(batch_num != num_batches - 1):
                        batch_indices = indices[batch_num*batch_size:(batch_num + 1)*batch_size]
                    else:
                        batch_indices = indices[batch_num*batch_size:]

                    # Generate batch
                    batch = (arr[batch_indices] for arr in (obs, returns, dones, actions, values, neg_log_probs))

                    # Run train step on batch
                    loss, pg_loss, value_loss, entropy, approx_kl = self.train_step(clip_range, entropy_coef, value_coef, *batch)

                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)

                    # Run optimizer step
                    self.policy_optim.step()

            # Tensorboard
            if(summary_writer is not None):

                summary_writer.add_scalar('Loss/total', loss, update*n_steps)
                summary_writer.add_scalar('Loss/policy', pg_loss, update*n_steps)
                summary_writer.add_scalar('Loss/value', value_loss, update*n_steps)
                summary_writer.add_scalar('Metrics/entropy', entropy, update*n_steps)
                summary_writer.add_scalar('Metrics/approx_kl', approx_kl, update*n_steps)
                summary_writer.add_scalar('Metrics/max_reward', sum(info["episode_rewards"])/len(info["episode_rewards"]), update*n_steps)
                summary_writer.add_scalar('Metricshalt_per_episode', info["HALT"]/len(info["episode_rewards"]), step = update*n_steps)

            # Comet
            if(comet_experiment is not None):
                comet_experiment.log_metric('total_loss', loss, step = update*n_steps)
                comet_experiment.log_metric('policy_loss', pg_loss, step = update*n_steps)
                comet_experiment.log_metric('value_loss', value_loss, step = update*n_steps)
                comet_experiment.log_metric('entropy_loss', entropy, step = update*n_steps)
                comet_experiment.log_metric('approx_kl', approx_kl, step = update*n_steps)
                comet_experiment.log_metric('episode_reward', sum(info["episode_rewards"])/len(info["episode_rewards"]), step = update*n_steps)
                comet_experiment.log_metric('halt_per_episode', info["HALT"]/len(info["episode_rewards"]), step = update*n_steps)


    def loss(self, clip_range,
                    entropy_coef,
                    value_coef,
                    returns,
                    values,
                    neg_log_probs,
                    entropies,
                    advantages,
                    old_values,
                    old_neg_log_probs):
        """Calculates the loss for a batch of data

        Args:
            clip_range: Clip range for the value function and ratio in policy loss.
            entropy_coef: Coefficient for entropy loss in total loss function.
            value_coef: Coefficient for value loss in total loss function.
            returns: Estimated returns at each time step (n_steps, self.input_dim)
            values: Value estimate of current policy (n_steps, self.output_dim)
            neg_log_probs: Negative log probabilities of action for current policy (n_steps, self.output_dim)
            entropies: Entropy of output distribution for current policy (n_steps, self.output_dim)
            advantages: Advantage estimates for current policy (n_steps, self.output_dim)
            old_values: Value estimate of policy used to generate action at each time step (n_steps, self.output_dim)
            old_neg_log_probs: Negative log probabilities of actions for policy used to generate actions at each time step (n_steps, self.output_dim)

        Returns:
            loss: Total loss
            pg_loss: Policy loss
            value_loss: Value loss
            entropy_mean: Entropy loss
            approx_kl: Approximate KL Divergence between old and new policy
        """

        ## Entropy loss ##
        entropy_loss = entropies.mean()

        ## Value Loss ##
        # Clip value update
        values_clipped = old_values + torch.clamp(values - old_values, min=-clip_range, max=clip_range)

        value_loss1 = (values - returns)**2
        value_loss2 = (values_clipped - returns)**2

        value_loss = .5 * torch.mean(torch.max(value_loss1, value_loss2))

        ## Policy loss ##
        ratios = torch.exp(old_neg_log_probs - neg_log_probs)

        pg_losses1 = -advantages * ratios
        pg_losses2 = -advantages * torch.clamp(ratios, 1.0 - clip_range, 1.0 + clip_range)


        pg_loss = torch.mean(torch.max(pg_losses1, pg_losses2))

        approx_kl = 0.5 * torch.mean((neg_log_probs - old_neg_log_probs)**2)

        ## Total Loss ##
        loss = pg_loss - (entropy_loss * entropy_coef) + (value_loss * value_coef)

        return loss, pg_loss, value_loss, entropy_loss, approx_kl

    def save(self, save_dir):
        torch.save(self.policy.state_dict(), os.path.join(save_dir, "policy.pt"))

    def load(self, load_dir):
        self.policy.load_state_dict(torch.load(os.path.join(load_dir, "policy.pt")))

    def eval(self, obs):

        # For n in range number of steps
        with torch.set_grad_enabled(False):
            # Convert obs to torch tensor
            # if(not isinstance(env, FakeEnv)):
            obs = torch.from_numpy(obs.copy()).float().to(self.device)
            obs = torch.unsqueeze(obs, 0)

            # Choose action
            action, _, _, _ = self.forward(observation = obs)


        return torch.squeeze(action)
