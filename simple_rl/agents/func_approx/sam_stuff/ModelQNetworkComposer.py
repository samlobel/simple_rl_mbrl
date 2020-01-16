import numpy as np
import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pdb
from copy import deepcopy
import shutil
import os
import time
import argparse
import pickle

import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from simple_rl.agents.AgentClass import Agent
from simple_rl.agents.func_approx.ddpg.utils import compute_gradient_norm
# from simple_rl.agents.func_approx.sam_stuff.replay_buffer import ReplayBuffer
from simple_rl.agents.func_approx.sam_stuff.model import ConvQNetwork, DenseQNetwork
from simple_rl.agents.func_approx.sam_stuff.epsilon_schedule import *
from simple_rl.tasks.gym.GymMDPClass import GymMDP
from simple_rl.tasks.lunar_lander.LunarLanderMDPClass import LunarLanderMDP
from simple_rl.agents.func_approx.sam_stuff.RandomNetworkDistillationClass import RNDModel, RunningMeanStd

from simple_rl.agents.func_approx.sam_stuff.DQNAgentClass import DQNAgent
#from simple_rl.agents.func_approx.sam_stuff.models import DenseTransitionModel, DenseRewardModel



class Composer:
    """
    NOTE: For completeness, this really does need something like a termination function. Just like STEVE.
    """

    def __init__(self, *, q_agent, world_model, action_size, device):
        """
        I'm just not sure we want to use the termination network yet... I don't know if training it will be if we even want to use the termination network...
        """
        self.q_agent = q_agent
        self.world_model = world_model
        self.action_size = action_size
        self.device = device

    def create_targets_for_episodes(self, experiences, gamma=0.99):
        """
        This one essentially takes in a replay buffer. That way I can just iterate backwards.
        Okay, it's going to zip straight back. Pretty nice.
        experiences:
            list of named tuples with keys 
            ["state","action","reward","next_state", "done", "time_limit_truncated"] 
        """

        values = []
        last_value = 0
        for exp in reversed(experiences):
            if exp.done:
                if exp.time_limit_truncated:
                    
                    last_value = self.q_agent.get_value(exp.next_state)
                    # import ipdb; ipdb.set_trace()
                    # print('singer')
                else:
                    last_value = 0
            value = exp.reward + (gamma*last_value)
            values.append(value)
            last_value = value
        # zipped

        values.reverse() # In place, always confusing
        return values

    def create_bias_variance_covariance_from_data(self, experiences, num_rollouts, gamma=0.99):
        """
        The big boy!!!
        """
        states = [exp.state for exp in experiences]
        actions = [exp.action for exp in experiences]

        true_values = self.create_targets_for_episodes(experiences, gamma=gamma)
        all_estimates = self.create_rollout_matrix_from_data(
            states, actions, num_rollouts, true_values, gamma=gamma)

        # import ipdb; ipdb.set_trace()

        bias = self.calculate_bias_for_rollouts(all_estimates, true_values)
        variance = self.calculate_variance_for_rollouts(all_estimates, true_values)

        covariance = self.calculate_covariance_for_rollouts(all_estimates, true_values)

        return bias, variance, covariance
        # unbiased_estimates = all_estimates - 
        

        # pass

    def calculate_covariance_for_rollouts(self, all_estimates, true_values):
        num_rollouts = len(all_estimates[0])

        true_values = np.asarray(true_values)
        true_values = true_values.reshape(-1, 1) # So you can subtract it...
        # debiased_estimates = np.asarray(debiased_estimates)
        # difference_matrix = debiased_estimates - true_values # The mean should be very close to zero...
        # assert np.allclose(difference_matrix.mean(axis=0), 0) # Make sure we're using debiased..
        difference_matrix = all_estimates - true_values

        covariance = np.cov(difference_matrix.T)
        assert covariance.shape == (num_rollouts, num_rollouts), f"{covariance.shape}, {num_rollouts}"

        return covariance
        # print(f"shape of independent_variance is {independent_variance.shape}")
        # return independent_variance

    def create_rollout_matrix_from_data(self, states, actions, num_rollouts, true_values, gamma=0.99):
        """
        True values are passed through from the `create-targets_for_episode`
        But this may be from multiple episodes, so I don't wanna just call it with the episode
        input
        """
        all_estimates = []
        for s, a, v in zip(states, actions, true_values):
            estimates = self.get_value_for_rollouts(s, a, num_rollouts, gamma=gamma)
            all_estimates.append(estimates)

        # This is a 100 x 5 matrix or something.
        return all_estimates

    def calculate_bias_for_rollouts(self, all_estimates, true_values):
        """
        Arguments:
            all_estimates:
                All of the rollout values. Something like 5 values for each sample.
                Equivalent to `X`.
            true_values:
                All of the true target values. Equivalent to `Y`
        """
        true_values = np.asarray(true_values)
        true_values = true_values.reshape(-1, 1) # So you can subtract it...
        # true_values = np.expand_dims(true_values, -1) # So you can subtract it...
        all_estimates = np.asarray(all_estimates)
        difference_matrix = all_estimates - true_values
        bias = difference_matrix.mean(axis=0)  # This is what you have to SUBTRACT!
        # print(f"Size of bias is {bias}")
        return bias

    def calculate_variance_for_rollouts(self, all_estimates, true_values):
        """
        Alright, there's actually no need to debias first... I can just skip that part.
        """
        true_values = np.asarray(true_values)
        true_values = true_values.reshape(-1, 1) # So you can subtract it...
        # debiased_estimates = np.asarray(debiased_estimates)
        # difference_matrix = debiased_estimates - true_values # The mean should be very close to zero...
        # assert np.allclose(difference_matrix.mean(axis=0), 0) # Make sure we're using debiased..
        difference_matrix = all_estimates - true_values

        independent_variance = np.var(difference_matrix, axis=0)

        print(f"shape of independent_variance is {independent_variance.shape}")
        return independent_variance


    def get_bias_variance_scaled_estimate(self, state, first_action, num_rollouts, bias, variance, gamma=0.99):
        assert len(bias) == len(variance) == num_rollouts, f"{len(bias)} {len(variance)} {num_rollouts}" # I may relax this to greater, but whatever.
        assert all(v >= 0 for v in variance), variance
        rollout_values = self.get_value_for_rollouts(state, first_action, num_rollouts, gamma=gamma)
        debiased_rollout_values = rollout_values - bias
        inverse_variances = [1 / v for v in variance]
        corrected_estimate = sum(iv*rv for iv, rv in zip(inverse_variances, debiased_rollout_values)) / sum(inverse_variances)
        return corrected_estimate

        # Is there really any reason to include the bias term? It was important for calculating
        # the variance, but now that that's done, since variance is fixed and whatnot, it really
        # is just going to be adding a constant. But at least it was necessary for calculating
        # variance. NOPE!!!!!
        # Now that we have debiased values, we can combine them with the covariance thing!

    def get_bias_covariance_scaled_estimate(self, state, first_action, num_rollouts, bias, covariance, gamma=0.99):
        assert len(bias) == len(covariance) == len(covariance[0]) == num_rollouts, f"{len(bias)} {len(covariance)} {num_rollouts}" # I may relax this to greater, but whatever.
        # assert all(v >= 0 for v in covariance.flatten()), covariance
        rollout_values = self.get_value_for_rollouts(state, first_action, num_rollouts, gamma=gamma)
        debiased_rollout_values = rollout_values - bias

        cov_inv = np.linalg.inv(covariance)
        ones = np.ones((num_rollouts, 1), np.float32)
        weights = (ones.T @ cov_inv) / (ones.T @ cov_inv @ ones)
        weights = weights.flatten()
        assert np.isclose(np.sum(weights), 1.0, atol=1e-6)

        # import pdb; pdb.set_trace()

        # print(f"Weights: {weights}")

        corrected_estimates = sum(w*rv for w, rv in zip(weights, debiased_rollout_values))
        # print(corrected_estimates)
        return corrected_estimates

        # inverse_variances = [1 / v for v in variance]
        # corrected_estimate = sum(iv*rv for iv, rv in zip(inverse_variances, debiased_rollout_values)) / sum(inverse_variances)
        # return corrected_estimate



    def get_best_action_for_bias_variance(self, state, num_rollouts, bias, variance, gamma=0.99):
        def functional(first_action):
            return self.get_bias_variance_scaled_estimate(state, first_action, num_rollouts, bias, variance, gamma=gamma)

        return self._get_best_action_from_functional(functional)

    def get_best_action_for_bias_covariance(self, state, num_rollouts, bias, variance, gamma=0.99):
        def functional(first_action):
            return self.get_bias_covariance_scaled_estimate(state, first_action, num_rollouts, bias, variance, gamma=gamma)

        return self._get_best_action_from_functional(functional)
    

    def _get_best_action_from_functional(self, functional):
        """
        The functional is something that takes in an action, and returns a value.
        It'll be created by filling in all the other values of some function like
        get_td_lambda_estimates
        """
        action_values = [functional(i) for i in range(self.action_size)]
        return np.argmax(np.asarray(action_values))

    def get_best_action_td_lambda(self, state, num_rollouts, gamma=0.99, lam=0.5):
        """The functional thing makes it much more readable and reusable IMO"""
        # print(lam)
        def functional(action):
            # print(lam)
            return self.get_td_lambda_estimates(state, action, num_rollouts, gamma=gamma, lam=lam)

        return self._get_best_action_from_functional(functional)      

    def get_td_lambda_estimates(self, state, first_action, num_rollouts, gamma=0.99, lam=0.5):
        # print(lam)
        rollout_values = self.get_value_for_rollouts(state, first_action, num_rollouts, gamma=gamma)
        lambda_scalers = [lam**i for i in range(num_rollouts)]
        # print(lambda_scalers)
        normalizing_factor = sum(lambda_scalers)
        scaled_rollouts = [norm * rval for norm, rval in zip(rollout_values, lambda_scalers)]
        # print(scaled_rollouts)
        lamda_estimate = sum(scaled_rollouts) / normalizing_factor
        return lamda_estimate
    
    def get_value_for_rollouts(self, state, first_action, num_rollouts, gamma=0.99):
        # Termination function can just influence current_discount

        rollout_values = []
        with torch.no_grad():
            discounted_reward_so_far = 0
            current_discount = 1.0
            current_state = state
            next_action = first_action

            for i in range(num_rollouts):
                # print(f"Doing rollout {i}")
                # Get the q_value
                # state_value = self.get_softmax_value(self, state, self.temperature)
                q_value = self.q_agent.get_qvalue(current_state, next_action)

                model_prediction = self.world_model.get_prediction(current_state, next_action)

                # Calculate this rollout's value.
                ### We're updating it so that it takes termination into account
                rollout_value = discounted_reward_so_far + (current_discount * q_value)
                # rollout_value = discounted_reward_so_far + (current_discount * state_value)
                # rollout_value = discounted_reward_so_far + (chance_unterminated * current_discount * q_value)
                rollout_values.append(rollout_value.item())

                # Accumulate reward
                # r_value = self.r_network(current_state, next_action)
                discounted_reward_so_far += (model_prediction['reward'] * current_discount)

                # Now that we've added reward, update discount,
                current_discount *= gamma
                # Related: update chance untermiated

                current_discount *= (1 - model_prediction['termination'])

                # Set next state and action.
                current_state = model_prediction['next_state']
                # current_state = self.t_network(current_state, next_action)
                next_action = self.q_agent.get_best_action(current_state)
        
        return rollout_values


                
