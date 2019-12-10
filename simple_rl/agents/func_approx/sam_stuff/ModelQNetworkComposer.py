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
from simple_rl.agents.func_approx.sam_stuff.replay_buffer import ReplayBuffer
from simple_rl.agents.func_approx.sam_stuff.model import ConvQNetwork, DenseQNetwork
from simple_rl.agents.func_approx.sam_stuff.epsilon_schedule import *
from simple_rl.tasks.gym.GymMDPClass import GymMDP
from simple_rl.tasks.lunar_lander.LunarLanderMDPClass import LunarLanderMDP
from simple_rl.agents.func_approx.sam_stuff.RandomNetworkDistillationClass import RNDModel, RunningMeanStd

from simple_rl.agents.func_approx.sam_stuff.DQNAgentClass import DQNAgent
from simple_rl.agents.func_approx.sam_stuff.models import DenseTransitionModel, DenseRewardModel


class Composer:
    """
    NOTE: For completeness, this really does need something like a termination function. Just like STEVE.
    """

    def __init__(self, q_network, t_network, r_network, termination_network=None):
        """I'm just not sure if we even want to use the termination network..."""
        self.q_network = q_network
        self.t_network = t_network
        self.r_network = r_network
        self.termination_network = termination_network

    def get_value_for_rollouts(self, state, first_action, num_rollouts, gamma=0.99):
        rollout_values = []
        with torch.no_grad():
            discounted_reward_so_far = 0
            current_discount = 1.0
            current_state = state
            next_action = first_action

            # first_q = self.q_network.get_qvalue(state, first_action)
            for i in num_rollouts:
                # Get the q_value
                q_value = self.q_network.get_qvalue(current_state, next_action)
                # Calculate this rollout's value.
                rollout_value = discounted_reward_so_far + (current_discount * q_value)
                rollout_values.append(rollout_value)

                # Accumulate reward
                r_value = self.r_network(current_state, next_action)
                discounted_reward_so_far += (r_value * current_discount)

                # Now that we've added reward, update discount, 
                current_discount *= gamma

                # Figure out next state and action.
                current_state = self.t_network(current_state, next_action)
                next_action = self.q_network.get_best_action(current_state)
        
        return rollout_values


                






def get_value_for_rollouts(q_network, t_network, r_network, num_rollouts, first_action):
    """
    Tentatively, we want to make a policy that chooses actions appropriately. That means adjusting
    Q-values to reflect on-policy learning. So, we take a first action, and then proceed on-policy.
    """

    first_q = q_network.get_qvalue()

    pass