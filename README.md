# MOReL - Model-Based Offline Reinforcement Learning

This is an implementation of the work presented in https://arxiv.org/abs/2005.05951 (Kidambi et al.).

MOReL is an algorithm for model-based offline reinforcement learning. At a high level, MOReL learns a dynamics model of the environment and also estimates uncertainty in the dynamics model. To optimize a policy, we apply a modified reward function, that provides a strong penatly for entering state/action pairs that have high uncertainty in the dynamics model.
