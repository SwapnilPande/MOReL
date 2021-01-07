import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchviz import make_dot

import numpy as np

class PolicyNet(nn.Module):
    def __init__(self, input_dim,
                output_dim,
                n_neurons = 64,
                activation = nn.Tanh,
                distribution = torch.distributions.multivariate_normal.MultivariateNormal):
        # Validate inputs
        assert input_dim > 0
        assert output_dim > 0
        assert n_neurons > 0

        super(PolicyNet, self).__init__()

        # Store configuration parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_neurons = n_neurons
        self.distribution = distribution

        # Create layers for the net
        self.input_layer = nn.Linear(input_dim, n_neurons)
        self.h0 = nn.Linear(n_neurons, n_neurons)
        self.h0_act = activation()
        self.h1 = nn.Linear(n_neurons, n_neurons)
        self.h1_act = activation()
        self.output_layer = nn.Linear(n_neurons, output_dim)

        self.h0v = nn.Linear(n_neurons, n_neurons)
        self.h0_actv = activation()
        self.h1v = nn.Linear(n_neurons, n_neurons)
        self.h1_actv = activation()
        self.value_head = nn.Linear(n_neurons, 1)

        self.var = torch.nn.Parameter(torch.tensor([0.0,0.0]).cuda(), requires_grad = True)

        self.mean_activation = nn.Tanh()
        # self.var_activation = nn.Softplus()

    def forward(self, obs, action = None):

        i = self.input_layer(obs)
        x = self.h0(i)
        x = self.h0_act(x)
        x = self.h1(x)
        x = self.h1_act(x)
        action_logit = self.output_layer(x)
        mean = action_logit[:,0:self.output_dim]
        var_logits = action_logit[:,self.output_dim:]

        x = self.h0v(i)
        x = self.h0_actv(x)
        x = self.h1v(x)
        x = self.h1_actv(x)
        value = self.value_head(x)

        # print(self.var)
        # mean = self.mean_activation(mean_logits)
        # var = self.var_activation(self.var)
        var = torch.exp(self.var)

        action_dist = self.distribution(mean, torch.diag_embed(var))


        if action is None:
            action = action_dist.sample()



        neg_log_prob = action_dist.log_prob(action) * -1.
        # print(neg_log_prob)
        entropy = action_dist.entropy()
        # print(entropy)


        return action, neg_log_prob, entropy, value

class ValueNet(nn.Module):
    def __init__(self, input_dim, n_neurons = 512, activation = nn.ReLU):
        # Validate inputs
        assert input_dim > 0
        assert n_neurons > 0

        super(ValueNet, self).__init__()

        # Store configuration parameters
        self.input_dim = input_dim
        self.n_neurons = n_neurons

        # Create layers for the net
        self.input_layer = nn.Linear(input_dim, n_neurons)
        self.h0 = nn.Linear(n_neurons, n_neurons)
        self.h0_act = activation()
        self.h1 = nn.Linear(n_neurons, n_neurons)
        self.h1_act = activation()
        self.output_layer = nn.Linear(n_neurons, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.h0(x)
        x = self.h0_act(x)
        x = self.h1(x)
        x = self.h1_act(x)
        x = self.output_layer(x)

        return x

class PPO2():
    def __init__(self, input_dim, output_dim, cuda = True):

        # Instantiate Networks
        if(cuda):
            self.policy_net = PolicyNet(input_dim, output_dim, n_neurons=64).cuda()
            # self.value_net = ValueNet(input_dim).cuda()
        else:
            self.policy_net = PolicyNet(input_dim, output_dim)
            # self.value_net = ValueNet(input_dim)

    def forward(self, observation, action = None):
        observation = torch.tensor(observation).float().cuda()

        action, neg_log_prob, entropy, value = self.policy_net(observation, action = action)

        #value = self.value_net(observation)

        return action, neg_log_prob, entropy, value

    def generate_experience(self, env, n_steps, gamma, lam):
        # Create observation vector
        obs = np.expand_dims(env.reset(), 0)
        done = False

        # Initialize memory buffer
        mb_obs, mb_rewards, mb_actions, mb_values, mb_done, mb_neg_log_prob = [],[],[],[],[],[]
        ep_infos = []
        total_reward = 0
        # For n in range number of steps
        with torch.set_grad_enabled(False):
            for i in range(n_steps):
                if(done):
                    obs = env.reset()
                    obs = np.expand_dims(obs, 0)
                    ep_infos.append(total_reward)

                # Choose action
                action, neg_log_prob, _, value = self.forward(obs)

                # Retrieve values
                action = np.squeeze(action.cpu().numpy())
                neg_log_prob = np.squeeze(neg_log_prob.cpu().numpy())
                value = np.squeeze(value.cpu().numpy())



                # Append data from step to memory buffer
                mb_obs.append(obs.copy())
                mb_actions.append(action)
                mb_values.append(value)
                mb_neg_log_prob.append(neg_log_prob)
                mb_done.append(done)

                # Step the environment, get new observation and reward
                obs, rewards, done, info = env.step(action)
                obs = np.expand_dims(obs, 0)
                total_reward += rewards


                # Append reward to memory buffer as well
                mb_rewards.append(rewards)


            # Convert memory buffer lists to numpy arrays
            mb_obs = np.concatenate(mb_obs, 0).astype(np.float32)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
            mb_actions = np.asarray(mb_actions)
            mb_values = np.asarray(mb_values, dtype=np.float32)
            mb_neg_log_prob = np.asarray(mb_neg_log_prob, dtype=np.float32)
            mb_done = np.asarray(mb_done, dtype=np.bool)

            # get value function for last action
            _, _, _, last_value = self.forward(obs)
            last_value = last_value.cpu().numpy()

            # Compute generalized advantage estimate by bootstrapping
            mb_advs = np.zeros_like(mb_rewards)
            last_gae_lam = 0
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
        return mb_rewards, mb_obs, mb_returns, mb_done, mb_actions, mb_values, mb_neg_log_prob, ep_infos

    def train_step(self, clip_range,
                        entropy_coef,
                        value_coef,
                        obs,
                        returns,
                        dones,
                        old_actions,
                        old_values,
                        old_neg_log_probs):
        # Create torch tensors and send to correct device
        # obs = torch.tensor(obs).float().cuda()
        returns = torch.tensor(returns).float().cuda()
        old_actions = torch.tensor(old_actions).float().cuda()
        old_values = torch.tensor(old_values).float().cuda()
        old_neg_log_probs = torch.tensor(old_neg_log_probs).float().cuda()

        # Calculate and normalize the advantages
        with torch.set_grad_enabled(False):
            advantages = returns - old_values

            # Normalize the advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.policy_net.train()
        # self.value_net.train()
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
            # print(loss)
            loss.backward()

            return loss, pg_loss, value_loss, entropy_mean, approx_kl




    def train(self, env, experiment, optimizer = torch.optim.Adam,
                                    lr =  0.00025,
                                    n_steps = 1024,
                                    time_steps = 1e6,
                                    clip_range = 0.2,
                                    entropy_coef = 0.01,
                                    value_coef = 0.5,
                                    batch_size = 256,
                                    gamma = 0.99,
                                    lam = 0.95,
                                    max_grad_norm = 0.5,
                                    num_train_epochs = 4,
                                    summary_writer = None):

        # Report multiple hyperparameters using a dictionary:
        hyper_params = {
            "learning_rate":  lr,
            "steps": 100000,
            "n_steps (simulation)": n_steps,
            "clip_range" : clip_range,
            "max_grad_norm" : max_grad_norm,
            "batch_size" : batch_size,
            "gamma" : gamma,
            "lam" : lam,
            "num_train_epochs" : num_train_epochs,
            "value_coef" : value_coef,
            "entropy_coef" : entropy_coef
        }
        experiment.log_parameters(hyper_params)

        n_updates = int(time_steps // n_steps)

        self.policy_optim = optimizer(self.policy_net.parameters(), lr = lr)
        # self.value_optim = optimizer(self.value_net.parameters(), lr = lr)

        # main train loop
        for update in range(n_updates):
            # Collect new experiences using the current policy
            rewards, obs, returns, dones, actions, values, neg_log_probs, infos = self.generate_experience(env, n_steps, gamma, lam)
            indices = np.arange(n_steps)
            for i in range(num_train_epochs):
                # Shuffle order of data
                np.random.shuffle(indices)

                num_batches = n_steps//batch_size
                if(n_steps % batch_size != 0):
                    num_batches += 1

                for batch_num in range(num_batches):
                    # Reset gradients
                    self.policy_net.zero_grad()
                    # self.value_net.zero_grad()

                    if(batch_num != num_batches - 1):
                        batch_indices = indices[batch_num*batch_size:(batch_num + 1)*batch_size]
                    else:
                        batch_indices = indices[batch_num*batch_size:]

                    batch = (arr[batch_indices] for arr in (obs, returns, dones, actions, values, neg_log_probs))
                    loss, pg_loss, value_loss, entropy, approx_kl = self.train_step(clip_range, entropy_coef, value_coef, *batch)

                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_grad_norm)
                    # torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_grad_norm)

                    self.policy_optim.step()

                    # self.value_optim.step()

            # Tensorboard
            if(summary_writer is not None):

                summary_writer.add_scalar('Loss/total', loss, update*n_steps)
                summary_writer.add_scalar('Loss/policy', pg_loss, update*n_steps)
                summary_writer.add_scalar('Loss/value', value_loss, update*n_steps)
                summary_writer.add_scalar('Metrics/entropy', entropy, update*n_steps)
                summary_writer.add_scalar('Metrics/approx_kl', approx_kl, update*n_steps)
                summary_writer.add_scalar('Metrics/max_reward', sum(infos)/len(infos), update*n_steps)

            experiment.log_metric('total_loss', loss, step = update*n_steps)
            experiment.log_metric('policy_loss', pg_loss, step = update*n_steps)
            experiment.log_metric('value_loss', value_loss, step = update*n_steps)
            experiment.log_metric('entropy_loss', entropy, step = update*n_steps)
            experiment.log_metric('approx_kl', approx_kl, step = update*n_steps)
            experiment.log_metric('episode_reward', sum(infos)/len(infos), step = update*n_steps)





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

        ## Entropy loss ##
        entropy_loss = entropies.mean()

        ## Value Loss ##
        # Clip value update
        values_clipped = old_values + torch.clamp(values - old_values, min=-clip_range, max=clip_range)

        value_loss1 = (values - returns)**2
        value_loss2 = (values_clipped - returns)**2

        # value_loss = F.smooth_l1_loss(value_f, reward)
        value_loss = .5 * torch.mean(torch.max(value_loss1, value_loss2))

        ## Policy loss ##
        ratios = torch.exp(old_neg_log_probs - neg_log_probs)

        # print("advantages {}".format(advantages))
        # print("mean {}".format(advantages.mean()))
        # print("std {}".format(advantages.std()))
        # print("ratios {}".format(ratios))
        pg_losses1 = -advantages * ratios
        pg_losses2 = -advantages * torch.clamp(ratios, 1.0 - clip_range, 1.0 + clip_range)
        # print("loss 1 m{}".format(pg_losses1.mean()))
        # print("loss 1 s{}".format(pg_losses1.std()))
        # print("loss_2 m{}".format(pg_losses2.mean()))
        # print("loss_2 s{}".format(pg_losses2.std()))
        # print("ratio m{}".format(ratios.mean()))
        # print("ratio s{}".format(ratios.std()))
        # print("neg_log_probs {}".format(old_neg_log_probs - neg_log_probs))
        # print("loss_m {}".format(torch.max(pg_losses1, pg_losses2)))


        pg_loss = torch.mean(torch.max(pg_losses1, pg_losses2))

        # print("loss {}".format(pg_loss))
        approx_kl = 0.5 * torch.mean((neg_log_probs - old_neg_log_probs)**2)
        # clip_frac = (torch.abs(ratio - 1.0) > clip_range).float().mean()

        loss = pg_loss - (entropy_loss * entropy_coef) + (value_loss * value_coef)

        # make_dot(loss, params = dict(self.policy_net.named_parameters())).render()

        return loss, pg_loss, value_loss, entropy_loss, approx_kl

