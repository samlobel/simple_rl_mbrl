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
        """I'm just not sure we want to use the termination network yet... I don't know if training it will be if we even want to use the termination network..."""
        self.q_agent = q_agent
        self.world_model = world_model
        self.action_size = action_size
        self.device = device

    def create_targets_for_episode(self, *, states, actions, rewards, true_finish=True, gamma=0.99, last_next_state):
        """
        It's a little confusing because in some sense, if something finishes because
        its out of time, we don't want to incorporate that into the MDP... Not sure
        exactly what to do there.

        We're getting the TRUE q values for all of these things...

        Just start from the end and add up?

        Should there be one more state than rewards? Because of "next state?" Not sure.
        Probably honestly. It's our best estimate of value at least.
        """
        if not true_finish:
            if last_next_state is None:
                raise Exception("If it's not a true finish, you need the next state's value!")
            last_value_estimate = max(self.q_agent.get_qvalues(last_next_state))
            # I NEED TO MAKE THIS PYTHONED.
        else:
            last_value_estimate = 0

        sar_tuples = list(zip(states, actions, rewards))
        sar_tuples_reversed = list(reversed(sar_tuples))

        values = []
        for state, action, reward in sar_tuples_reversed:
            v = reward + (gamma * last_value_estimate)
            values.append(v)
            last_value_estimate = v

        values.reverse() # Modifies in place... dangerous

        # Whoever gave this to us should probably have the values themselves.
        return values

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
        true_values = np.asarray(true_values)
        true_values = true_values.reshape(-1, 1) # So you can subtract it...
        # true_values = np.expand_dims(true_values, -1) # So you can subtract it...
        all_estimates = np.asarray(all_estimates)
        difference_matrix = all_estimates - true_values
        bias = difference_matrix.mean(axis=0)  # This is what you have to SUBTRACT!
        print(f"Size of bias is {bias}")
        return bias

    def calculate_variance_for_debiased_rollouts(self, debiased_estimates, true_values):
        true_values = np.asarray(true_values)
        true_values = true_values.reshape(-1, 1) # So you can subtract it...
        debiased_estimates = np.asarray(debiased_estimates)
        difference_matrix = debiased_estimates - true_values # The mean should be very close to zero...
        assert np.allclose(difference_matrix.mean(axis=0), 0) # Make sure we're using debiased..
        independent_variance = np.var(difference_matrix, axis=0)
        print(f"shape of independent_variance is {independent_variance.shape}")
        return independent_variance




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
        def functional(action):
            return self.get_td_lambda_estimates(state, action, num_rollouts, gamma=gamma, lam=lam)

        return self._get_best_action_from_functional(functional)      

    def get_td_lambda_estimates(self, state, first_action, num_rollouts, gamma=0.99, lam=0.5):
        rollout_values = self.get_value_for_rollouts(state, first_action, num_rollouts, gamma=gamma)
        lambda_scalers = [lam**i for i in range(num_rollouts)]
        normalizing_factor = sum(lambda_scalers)
        scaled_rollouts = [norm * rval for norm, rval in zip(rollout_values, lambda_scalers)]
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
                q_value = self.q_agent.get_qvalue(current_state, next_action)

                model_prediction = self.world_model.get_prediction(current_state, next_action)

                # Calculate this rollout's value.
                ### We're updating it so that it takes termination into account
                rollout_value = discounted_reward_so_far + (current_discount * q_value)
                # rollout_value = discounted_reward_so_far + (chance_unterminated * current_discount * q_value)
                rollout_values.append(rollout_value)

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


                
