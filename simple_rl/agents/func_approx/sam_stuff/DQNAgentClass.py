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

from termcolor import colored, cprint


from simple_rl.agents.AgentClass import Agent
from simple_rl.agents.func_approx.ddpg.utils import compute_gradient_norm
from simple_rl.agents.func_approx.sam_stuff.replay_buffer import ReplayBuffer
from simple_rl.agents.func_approx.sam_stuff.model import ConvQNetwork, DenseQNetwork
from simple_rl.agents.func_approx.sam_stuff.epsilon_schedule import *
from simple_rl.tasks.gym.GymMDPClass import GymMDP
# from simple_rl.tasks.lunar_lander.LunarLanderMDPClass import LunarLanderMDP

from simple_rl.agents.func_approx.sam_stuff.model import DenseTransitionModel, DenseRewardModel, DenseTerminationModel


## Hyperparameters
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 32  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
# TAU = 1e-1  # for soft update of target parameters
LR = 1e-4  # learning rate
UPDATE_EVERY = 1  # how often to update the network
# UPDATE_EVERY = 10  # how often to update the network
NUM_EPISODES = 3500
NUM_STEPS = 10000

LOG_EVERY=10

# TODO: I should add an option to use a central replay-buffer. That way, we don't
# need to keep like a million of them around that all have the same data. Also that way,
# I can control how things get passed in from one central location. 

# TODO: Do I need to add a empirical-Q entry to the replay buffer? No, I'll just
# do those updates totally within the online Q composer, no buffer necessary.
# That's annoying, that means I need to do it only at the end of an episode. Oh well.
# I need to do constant covariance updates I think, because our model and Q-functions
# are always going to be changing as well.

# TODO: I need to implement/copy a SAC model into here. It honestly doesn't seem all
# that bad to do myself. But also, there's a lot I can do myself before then. In
# cartpole I can just do a DQN with soft actions, and a decaying gamma.



class WorldModel(nn.Module):
    """Sort of like DQNAgent for the world-model.
    This should be helpful because we can use it to train all the things
    at once.
    """
    def __init__(self, state_size, action_size,
                 seed, device, name="WorldModel",
                 # eps_start=1.,
                 tensor_log=False, lr=LR,
                 # use_double_dqn=True,
                 gamma=GAMMA,
                 # loss_function="huber",
                 gradient_clip=None,
                 #evaluation_epsilon=0.05, exploration_method="eps-greedy",
                 pixel_observation=False, writer=None,
                 epsilon_linear_decay=100000):

        nn.Module.__init__(self)

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = lr
        self.gamma = gamma
        # I'm going to just do MSE for losses.
        self.gradient_clip = gradient_clip
        self.pixel_observation = pixel_observation
        self.seed = random.seed(seed)
        np.random.seed(seed)
        self.tensor_log = tensor_log
        self.device = device

        # Q-Network
        if pixel_observation:
            raise Exception("World model doesn't work with pixel_observation right now but leaving it in in case I get serious")
            # self.policy_network = ConvQNetwork(in_channels=4, n_actions=action_size).to(self.device)
            # self.target_network = ConvQNetwork(in_channels=4, n_actions=action_size).to(self.device)
        else:
            self.transition_model = DenseTransitionModel(state_size, action_size, seed).to(self.device)
            self.reward_model = DenseRewardModel(state_size, action_size, seed).to(self.device)
            self.termination_model = DenseTerminationModel(state_size, action_size, seed).to(self.device)
            # self.policy_network = DenseQNetwork(state_size, action_size, seed, fc1_units=32, fc2_units=16).to(self.device)
            # self.target_network = DenseQNetwork(state_size, action_size, seed, fc1_units=32, fc2_units=16).to(self.device)

        all_parameters = list(self.transition_model.parameters()) + list(self.reward_model.parameters()) + list(self.termination_model.parameters())
        self.optimizer = optim.Adam(all_parameters, lr=lr)
        # Replay memory
        self.replay_buffer = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, self.device, pixel_observation)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        self.num_executions = 0 # Number of times act() is called (used for eps-decay)

        # Debugging attributes
        self.num_updates = 0
        self.num_epsilon_updates = 0

        if self.tensor_log:
            self.writer = SummaryWriter() if writer is None else writer

        print("\nCreating {} with lr={} and buffer_sz={}\n".format(name, self.learning_rate, BUFFER_SIZE))


    def forward(self, *args, **kwargs):
        pass

    def get_prediction(self, state, action):
        """
        Unfortunately, needs to handle numpy arrays AND torch things.
        AND ints/lazy-states.
        """
        # Should work in similar circumstances to "dqn_agent.act"
        # We're assuming 'action' is a simple integer...
        # Instead we need it to be an array

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        elif isinstance(state, torch.Tensor):
            pass
        else:
            state = torch.from_numpy(np.asarray(state))

        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        elif isinstance(action, torch.Tensor):
            pass
        else:
            action = torch.from_numpy(np.asarray(action))

        state = state.float().unsqueeze(0).to(self.device)
        action = action.long().unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            next_state = self.transition_model(state, action).squeeze()
            reward = self.reward_model(state, action).squeeze()
            termination = self.termination_model(state, action, mode="probs").squeeze()


        return dict(
            next_state=next_state,
            reward=reward,
            termination=termination
        )

    def step(self, state, action, reward, next_state, done, num_steps=1, time_limit_truncated=False):
        self.replay_buffer.add(state, action, reward, next_state, done, num_steps, time_limit_truncated)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.replay_buffer) > BATCH_SIZE:
                experiences = self.replay_buffer.sample(batch_size=BATCH_SIZE)
                # self._learn(experiences, GAMMA)
                self._learn(experiences)
                if self.tensor_log:
                    self.writer.add_scalar("NumPositiveTransitionsWorldModel", self.replay_buffer.positive_transitions[-1], self.num_updates)
                self.num_updates += 1
        # raise NotImplementedError()

    def _learn(self, experiences):#, gamma):
        """
        Update value parameters using given batch of experience tuples.
        Args:
            experiences (tuple<torch.Tensor>): tuple of (s, a, r, s', done, tau) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, steps, time_limit_truncateds = experiences

        # To get the "done" ones that really count, it needs to be done and NOT been time limit truncated.
        truly_dones = dones * (1 - time_limit_truncateds)

        next_states_predicted = self.transition_model(states, actions)
        rewards_predicted = self.reward_model(states, actions)
        dones_predicted = self.termination_model(states, actions, mode="logits")

        transition_error = F.mse_loss(next_states, next_states_predicted)
        reward_error = F.mse_loss(rewards, rewards_predicted)
        termination_error = F.binary_cross_entropy_with_logits(dones_predicted, truly_dones)

        loss = transition_error + reward_error + termination_error # They're independent networks...
        self.optimizer.zero_grad()
        loss.backward()

        if self.gradient_clip is not None:
            for param in self.all_parameters:
                param.grad.data.clamp_(-self.gradient_clip, self.gradient_clip)

        self.optimizer.step()

        # if self.tensor_log:
        if self.tensor_log and self.num_updates % LOG_EVERY == 0:
            # print('logging...')
            # print(self.num_updates)
            self.writer.add_scalar("Total-WorldModel-Loss", loss.item(), self.num_updates)
            self.writer.add_scalar("Transition-Prediction-Loss", transition_error.item(), self.num_updates)
            self.writer.add_scalar("Reward-Prediction-Loss", reward_error.item(), self.num_updates)
            self.writer.add_scalar("Termination-Prediction-Loss", termination_error.item(), self.num_updates)

            self.writer.add_scalar("TransFunc-GradientNorm", compute_gradient_norm(self.transition_model), self.num_updates)
            self.writer.add_scalar("RewFunc-GradientNorm", compute_gradient_norm(self.reward_model), self.num_updates)
            self.writer.add_scalar("TermFunc-GradientNorm", compute_gradient_norm(self.termination_model), self.num_updates)


class DQNAgent(Agent, nn.Module):
    """Interacts with and learns from the environment.
    
    Why do I add the "module" part? I really don't remember.

    Policy Network:
        Chooses actions
    Target Network:
        Values states for action-updates.
    Policy network is the one that's updated directly. Just so I know.
    """

    def __init__(self, state_size, action_size,
                 seed, device, name="DQN-Agent",
                 eps_start=1., tensor_log=False, lr=LR, tau=TAU, use_double_dqn=True, gamma=GAMMA,
                 loss_function="huber",
                 gradient_clip=None, evaluation_epsilon=0.05, exploration_method="eps-greedy",
                 pixel_observation=False, writer=None,
                 use_softmax_target=False,
                 softmax_temperature=0.1,
                 epsilon_linear_decay=100000):
        nn.Module.__init__(self)

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = lr
        self.tau = tau
        self.use_ddqn = use_double_dqn
        self.gamma = gamma
        self.loss_function = loss_function
        self.gradient_clip = gradient_clip
        self.evaluation_epsilon = evaluation_epsilon
        self.exploration_method = exploration_method
        self.pixel_observation = pixel_observation
        self.seed = random.seed(seed)
        np.random.seed(seed)
        self.tensor_log = tensor_log
        self.device = device
        self.use_softmax_target = use_softmax_target
        if self.use_softmax_target:
            self.temperature = softmax_temperature # Why the heck not.

        # Q-Network
        if pixel_observation:
            self.policy_network = ConvQNetwork(in_channels=4, n_actions=action_size).to(self.device)
            self.target_network = ConvQNetwork(in_channels=4, n_actions=action_size).to(self.device)
        else:
            self.policy_network = DenseQNetwork(state_size, action_size, seed, fc1_units=32, fc2_units=16).to(self.device)
            self.target_network = DenseQNetwork(state_size, action_size, seed, fc1_units=32, fc2_units=16).to(self.device)

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

        # Replay memory
        self.replay_buffer = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, self.device, pixel_observation)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # Epsilon strategy
        if exploration_method == "eps-greedy":
            self.epsilon_schedule = GlobalEpsilonSchedule(eps_start, evaluation_epsilon, eps_lin_dec=epsilon_linear_decay) if "global" in name.lower() else OptionEpsilonSchedule(eps_start)
            self.epsilon = eps_start
        else:
            raise NotImplementedError("{} not implemented", exploration_method)

        self.num_executions = 0 # Number of times act() is called (used for eps-decay)

        # Debugging attributes
        self.num_updates = 0
        self.num_epsilon_updates = 0

        if self.tensor_log:
            self.writer = SummaryWriter() if writer is None else writer

        print("\nCreating {} with lr={} and ddqn={} and buffer_sz={}\n".format(name, self.learning_rate,
                                                                               self.use_ddqn, BUFFER_SIZE))

        Agent.__init__(self, name, range(action_size), GAMMA)

    def act(self, state, train_mode=True):
        """
        Interface to the DQN agent: state can be output of env.step() and returned action can be input into next step().
        Args:
            state (np.array): numpy array state from Gym env
            train_mode (bool): if training, use the internal epsilon. If evaluating, set epsilon to min epsilon

        Returns:
            action (int): integer representing the action to take in the Gym env
        """
        if train_mode == False and self.use_softmax_target:
            print("Probably something wrong going on here, contradictory options")
            raise Exception("Contraaaadicion!")

        self.num_executions += 1
        epsilon = self.epsilon if train_mode else self.evaluation_epsilon

        state = np.array(state)  # Lazy Frame
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.policy_network.eval()

        if self.use_softmax_target:
            with torch.no_grad():
                chosen_action = self.get_softmax_action(state, self.temperature)
            return chosen_action.item()


        with torch.no_grad():
            action_values = self.policy_network(state)

        self.policy_network.train()

        action_values = action_values.cpu().data.numpy()

        # Epsilon-greedy action selection
        if random.random() > epsilon:
            return np.argmax(action_values)

        return random.choice(list(set(self.actions)))

    def forward(self, *args, **kwargs):
        print("This probably shouldn't be called, I'm mainly including it so that we can have the save option work.")
        to_return = self.act(self, *args, **kwargs)
        print("Called the act method as a submethod, returning now.")
        return to_return

    def get_best_action(self, state):
        """
        This was written by Sam, meaning you just can't trust it.
        This should really be returning a SINGLE VALUE... It returns a torch
        tensor right now.
        """
        assert len(state.shape) == 1

        q_values = self.get_qvalues(state, network_name="policy")
        # to_return = torch.argmax(q_values).item()
        # print(f"{to_return}   {q_values}")
        # return to_return
        return torch.argmax(q_values).item()

    def get_value(self, state):
        """
        Gets the "true" value for the state, meaning what would happen if you took the
        best action from here.

        THIS ONLY WORKS IF ITS A SINGLE STATE!!!!!
        """
        print("Ideally, I would like this to work with either one or many samples," +
              " and just return in kind")
        cprint("\nThis shuold only be used in the deprecated composer anyways...\n", "red")
        action_values = self.get_qvalues(state)

        # Argmax only over actions that can be implemented from the current state
        return np.max(action_values.cpu().data.numpy())

    def get_softmax_values(self, state, temperature=None):
        """
        This takes actions proportionately to their value
        I would really like all of these methods to return the reasonable shape,
        regardless their input shape.

        If it's ddqn, this should really be using the target network for
        the value part, and the policy network for the action-values
        part....

        It just does ddqn implicitly. I'll
        just leave it at that for the time being.


        """
        if temperature is None:
            temperature = self.temperature

        action_values_for_prob = self.get_qvalues(state, network_name="policy")
        tempered_action_values_for_prob = action_values_for_prob / temperature
        # print('really not sure about the dim, it should work for all sizes though.')
        probability = F.softmax(tempered_action_values_for_prob, dim=-1)

        action_values_for_value = self.get_qvalues(state, network_name="target")
        # action_values_for_value = self.get_qvalues(state, network_name="policy")

        weighted_values = action_values_for_value * probability # elem-wise
        summed_weighted_values = weighted_values.sum(dim=-1)

        return summed_weighted_values
        # return summed_weighted_values.cpu().data.numpy()

    def get_softmax_action(self, state, temperature=None):
        """
        I think I can use torch.multinomial for this...
        I should be using the policy network for this... because it's action selection.

        This one really needs to be able to work regardless of dimensionality
        """
        if temperature is None:
            temperature = self.temperature

        action_values = self.get_qvalues(state, network_name="policy").squeeze(0)

        temperatured_action_values = action_values / temperature
        probability = F.softmax(temperatured_action_values, dim=-1)
        return torch.multinomial(probability, num_samples=1).unsqueeze(-1)

    def get_qvalue(self, state, action_idx):
        # This one sort of makes sense it's only one state. Although,
        # I guess it instead could use 'gather'.
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        state = state.float().unsqueeze(0).to(self.device)
        self.policy_network.eval()
        with torch.no_grad():
            action_values = self.policy_network(state)
        self.policy_network.train()
        return action_values[0][action_idx]
    
    def _choose_network(self, network_name):
        if network_name == "policy":
            return self.policy_network
        elif network_name == "target":
            return self.target_network
        else:
            raise Exception(f"Expected either target or policy but got {network_name}")

    def get_qvalues(self, state, network_name="policy"):
        """
        This should allow either 1 or 2 dimensional inputs.
        
        It should also probably let you choose between policy and target...
        not sure though
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)

        singleton = False
        if len(state.shape) == 1:
            singleton = True
            state = state.unsqueeze(0)

        network = self._choose_network(network_name)

        state = state.float().to(self.device)
        # state = state.float().unsqueeze(0).to(self.device)
        # self.policy_network.eval()
        # with torch.no_grad():
        #     action_values = self.policy_network(state)
        # self.policy_network.train()
        network.eval()
        with torch.no_grad():
            action_values = network(state)
        network.train()

        if singleton:
            action_values = action_values.squeeze(0)

        return action_values

    def step(self, state, action, reward, next_state, done, num_steps=1, time_limit_truncated=False):
        """
        Interface method to perform 1 step of learning/optimization during training.
        Args:
            state (np.array): state of the underlying gym env
            action (int)
            reward (float)
            next_state (np.array)
            done (bool): is_terminal
            num_steps (int): number of steps taken by the option to terminate
        """
        # Save experience in replay memory
        self.replay_buffer.add(state, action, reward, next_state, done, num_steps, time_limit_truncated)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.replay_buffer) > BATCH_SIZE:
                experiences = self.replay_buffer.sample(batch_size=BATCH_SIZE)
                self._learn(experiences, GAMMA)
                if self.tensor_log:
                    self.writer.add_scalar("NumPositiveTransitions", self.replay_buffer.positive_transitions[-1], self.num_updates)
                self.num_updates += 1
                # if self.num_updates % 1000 == 0:
                #     print(f"\nNumber of updates: {self.num_updates}")

    def _get_state_values(self, states):
        """
        This is an easier one to overwrite than _learn...

        It's going to go inside of _learn. I feel like it should either use or replace
        a lot of the other ones.... why are there so many!

        """
        if self.use_ddqn:
            self.policy_network.eval()
            with torch.no_grad():
                if self.use_softmax_target:
                    state_values = self.get_softmax_values(states)
                    # This used to be very bad in that it was the wrong dimensionality.
                    state_values = state_values.unsqueeze(1)
                else:
                    selected_actions = self.policy_network(states).argmax(dim=1).unsqueeze(1)
                    state_values = self.target_network(states).detach().gather(1, selected_actions)
            self.policy_network.train()
            return state_values.detach()
        else:
            raise NotImplementedError("bahahahaha!!!!")
            # I feel like there's a whole lot of re-doing of turning off gradients.
            # That's a problem for later though.

    def _learn(self, experiences, gamma, Q_targets_next=None):
        """
        Update value parameters using given batch of experience tuples.
        Args:
            experiences (tuple<torch.Tensor>): tuple of (s, a, r, s', done, tau) tuples
            gamma (float): discount factor

        When we do this softmax thing, we should really be doing it with the policy network
        choosing actions and the target network giving values. For consistency.
        """
        states, actions, rewards, next_states, dones, steps, time_limit_truncateds = experiences

        Q_targets_next = self._get_state_values(next_states)

        # Options in SMDPs can take multiple steps to terminate, the Q-value needs to be discounted appropriately
        discount_factors = gamma ** steps

        # Compute Q targets for current states
        # Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Q_targets = rewards + (discount_factors * Q_targets_next * (1 - dones))
        # If you just timed out, you should still do the Q-targets_next thing.
        truly_dones = dones * (1 - time_limit_truncateds)
        Q_targets = rewards + (discount_factors * Q_targets_next * (1 - truly_dones))

        # Get expected Q values from local model
        Q_expected = self.policy_network(states).gather(1, actions)

        # Compute loss
        if self.loss_function == "huber":
            # print("Getting an error here?")
            # import pdb; pdb.set_trace()
            loss = F.smooth_l1_loss(Q_expected, Q_targets)
        elif self.loss_function == "mse":
            loss = F.mse_loss(Q_expected, Q_targets)
        else:
            raise NotImplementedError("{} loss function type not implemented".format(self.loss_function))

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()

        # # Gradient clipping: tried but the results looked worse -- needs more testing
        if self.gradient_clip is not None:
            for param in self.policy_network.parameters():
                param.grad.data.clamp_(-self.gradient_clip, self.gradient_clip)

        self.optimizer.step()

        # if self.tensor_log:
        if self.tensor_log and self.num_updates % LOG_EVERY == 0:
            # print("ENTERING DQN LOGGING....")
            self.writer.add_scalar("DQN-Loss", loss.item(), self.num_updates)
            self.writer.add_scalar("DQN-AverageTargetQvalue", Q_targets.mean().item(), self.num_updates)
            self.writer.add_scalar("DQN-AverageQValue", Q_expected.mean().item(), self.num_updates)
            self.writer.add_scalar("DQN-GradientNorm", compute_gradient_norm(self.policy_network), self.num_updates)

        # ------------------- update target network ------------------- #
        self.soft_update(self.policy_network, self.target_network, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """
        Args:
            local_model (nn.Module): weights will be copied from
            target_model (nn.Module): weights will be copied to
            tau (float): interpolation parameter - usually small eg 0.0001
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def update_epsilon(self):
        self.num_epsilon_updates += 1
        self.epsilon = self.epsilon_schedule.update_epsilon(self.epsilon, self.num_epsilon_updates)

        # if random.random() < 0.01:
        #     # This is certainly a silly way of doing this...
        #     print("We're going to make temperature and epsilon track")
        # self.temperature = self.epsilon

        # Log epsilon decay
        if self.tensor_log:
            self.writer.add_scalar("DQN-Epsilon", self.epsilon, self.num_epsilon_updates)


class OnlineComposer(DQNAgent):
    """

    I can definitely do the covariance thing incrementally. The only thing is, I need
    to use the running mean, not the sample-mean.

    This acts surprisingly similar to the Composer in ModelQNetworkComposer.py
    Except that it acts online, and uses the composer to provide better updates,
    as opposed to better evaluation. I guess it could do both....

    I think that I actually need to walk through an entire episode in order to add to the
    replay buffer here. The reason being, if I want the GROUND TRUTH values for the
    Q functions, we need an entire trajectory's data. That definitely changes things.
    I can expect that this is the thing that interacts directly with the environment.
    So we can just store a single episode on this.

    I don't know if I should use the q agent or just import the target agents directly...
    I'll start with the real Q agent.

    Action selection should be done exactly like we do. The difference is going to
    be that the TARGETS are calculated using our special method.
    The confusing thing is, that means overwriting the choosing function somehow. That's
    pretty annoying.

    I actually think that extending the q agent is the right thing to do here, surprisingly.
    That's because I would get to keep a lot of the stuff, I would mainly have to overwrite the
    "step" function. And add in world-model training and whatnot.

    I found a way to do it. Why, I have no idea. But it keeps track of a weight vector.
    So long as the weight vector stays positive, which it should.... then we can keep a
    running average and still expect it to behave properly. So, how to combine these things
    over time is an open question for now. In some sense, this should be bayesian.

    I think I should calculate a bias vector, combine it with the old one, then use THAT.
    Then, I can get a new covariance matrix, which I'll average with the old one (?).
    Finally, I'll calculate a whole new w from that process, and directly use that.
    If you look at the definition of covariance, this sort of works. Except its sort of like
    a block-diagonal approach. Because we get the cross-multipliers of things in the
    same batch, but not in different batches. Maybe we should hold on to more episodes?
    The reason to do it with some sort of back-off pattern is that old things get stale,
    as they're no longer accurate representations of the policy, OR of the true value.

    Ideally, what we'd do is, somehow bake that in to the normalization constant.
    I like that a lot. But I don't really have a great way to do this, because I don't know
    statistics. What I would do is weight episodes geometrically decreasing, and hold on to
    like 10 of them. And then for covariance I would do a weighted average of them, instead
    of a regular average. But not for now, for now I'll do something simpler. Either,
    averaging weight vectors. OR, keeping a bunch of episodes around, and just replacing.
    I'll start with averaging weight vectors.

    Interestingly, one way to do this easily in a buffer would be: store the episode number.
    As well as the current episode number. And just geometrically discount it
    so that we value previous ones less. And use a buffer.
    """

    def __init__(self, *args, world_model=None, num_rollouts=5, mixing_speed=0.95, **kwargs):
        """
        Okay, how should we be initializing? We need a starting bias and covariance.
        We also need a mixing speed, and a way to do the averaging.
        """
        super(OnlineComposer, self).__init__(*args, **kwargs)

        self.world_model = world_model
        self.num_rollouts = num_rollouts

        self.episode_buffer = []
        # This should look familiar...
        self.episode_experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "num_steps", "time_limit_truncated"])

        self.biases = np.zeros((num_rollouts,))
        self.covariances = np.eye(num_rollouts, dtype=np.float32)

        self.weights = np.ones((num_rollouts)) / num_rollouts # Let's start with just the Q

        self.mixing_speed = mixing_speed #0.95 # Something like a half life of 14....


    # def forward(self, *args, **kwargs):
    #     pass

    def step(self, state, action, reward, next_state, done, num_steps=1, time_limit_truncated=False):
        exp = self.episode_experience(state, action, reward, next_state, done, num_steps, time_limit_truncated)
        self.episode_buffer.append(exp)
        if exp.done:
            print("Updating weight!")
            self._update_weights(self.episode_buffer)
            self.episode_buffer = []
            print(self.weights)
        self.world_model.step(state, action, reward, next_state, done, num_steps=1, time_limit_truncated=False)
        # This step calls "learn" sometimes. That's good, we need to overwrite that though....
        super(OnlineComposer, self).step(state, action, reward, next_state, done, num_steps=num_steps, time_limit_truncated=time_limit_truncated)
        pass
    
    # def _learn(self, experiences):
    #     print(f"Learning using weights {self.weights} and biases {self.biases}")
    #     pass

    def _get_state_values(self, states):
        """
        This is actually all I need to update if I
        want to replace the existing update with this.

        Really make sure that it's 32 x 1 or whatever.

        NOTE: This returns a numpy array, but that should maybe be alright given how it's used.
        """
        print('bingster!!!!')
        import pdb; pdb.set_trace()
        all_estimates = self._get_values_for_rollouts(states)
        debiased_estimates = all_estimates - self.biases # This may need some broadcast support
        weighted_estimates = all_estimates @ self.weights.T
        # weighted_estimates = np.squeeze(weighted_estimates, axis=1)
        return weighted_estimates

    # def _get_weighted_state_values(self, states):
    #     # First, you get the rollout values.
    #     all_estimates = self._get_values_for_rollouts(states)
    #     debiased_estimates = all_estimates - self.biases # This may need some broadcast support
    #     weighted_estimates = all_estimates @ self.weights.T
    #     # weighted_estimates = np.squeeze(weighted_estimates, axis=1)
    #     return weighted_estimates

    # def _create_bias_covariance_from_data(self, experiences, gamma=0.99):
    #     """
    #     The big boy!!!
    #     """

    #     num_rollouts = self.num_rollouts

    #     states = [exp.state for exp in experiences]
    #     actions = [exp.action for exp in experiences]

    #     true_values = self._get_true_state_values(experiences, gamma=gamma)
    #     all_estimates = self._get_values_for_rollouts(states)
    #     # all_estimates = self.create_rollout_matrix_from_data(
    #     #     states, actions, num_rollouts, true_values, gamma=gamma)

    #     # import ipdb; ipdb.set_trace()

    #     bias = self._calculate_bias_for_rollouts(all_estimates, true_values)
    #     covariance = self._calculate_covariance_for_rollouts(all_estimates, true_values)

    #     return bias, covariance

    def _get_true_state_values(self, episode, gamma):
        """
        It's a bit unfortunate that on episodes that end by time, we just peg
        it to the Q value. Because then that's going to dominate because
        it's perfect... I could peg it to the covariance-modified value,
        that might be best. That way, maybe on average it wouldn't change anything?
        """
        # Episode is a list of experience tuples... They should be named.
        """I must have done this somewhere else..."""
        # First, assert that the last one is "done"
        assert episode[-1].done == True, f"last one needs to be done, instead {episode[-1].done}"
        values = []
        for exp in reversed(episode):
            if exp.done:
                if exp.time_limit_truncated:
                    last_value = self.q_agent.get_softmax_value(exp.next_state)
                    # import ipdb; ipdb.set_trace()
                    # print('singer')
                else:
                    last_value = 0
            value = exp.reward + (gamma*last_value)
            values.append(value)
            last_value = value
        values.reverse()
        return np.asarray(values) # Should be fine.


    def _get_values_for_rollouts(self, states: torch.Tensor):
        """
        In a perfect world maybe we would decorrelate these. But on the
        other hand, the covariance matrix should take care of that, so long
        as we do the same thing at train and test time.

        We should be able to do a whole batch at once, why not right?
        That involves making it a torch tensor for sure. This will be called only
        internally, so we can make sure that's true elsewhere.

        """

        num_rollouts = self.num_rollouts
        gamma = self.gamma

        rollout_values = []
        assert len(states.shape) == 2
        batch_size = states.shape[0]
        # batch_size = len(states) # I think that's fine.



        with torch.no_grad():
            discounted_reward_so_far = torch.zeros((batch_size), dtype=torch.float32)
            current_discounts = torch.ones((batch_size), dtype=torch.float32)
            current_states = states
 
            # discounted_reward_so_far = 0
            # current_discount = 1.0
            # current_states = states
            # next_action = first_action

            for i in range(num_rollouts):
                # print(f"Doing rollout {i}")
                # Get the q_value
                q_value = self.get_softmax_value(current_states)
                next_actions = self.get_softmax_action(current_states, self.temperature)

                model_prediction = self.world_model.get_prediction(current_states, next_actions)

                # Calculate this rollout's value.
                ### We're updating it so that it takes termination into account
                rollout_value = discounted_reward_so_far + (current_discount * q_value)
                rollout_values.append(rollout_value)

                # Accumulate reward
                discounted_reward_so_far += (model_prediction['reward'] * current_discount)

                # Now that we've added reward, update discount,
                current_discount *= gamma
                # Related: update chance untermiated

                current_discount *= (1 - model_prediction['termination'])

                # Set next state and action.
                current_states = model_prediction['next_state']
                # current_state = self.t_network(current_state, next_action)
                # next_action = self.q_agent.get_best_action(current_state)
        
        return np.asarray(rollout_values)

    def _calculate_bias_for_rollouts(self, all_estimates, true_values):
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

    def _calculate_covariance_for_rollouts(self, all_estimates, true_values):
        """
        We're going to debias using the running average, and then calculate it ourself.

        NOTE: The estimates are not debiased yet.
        """
        cprint("Not really sure about this ATM","red")

        num_samples = len(all_estimates)

        # true values is the real Q value given the data. Should be unbiased.
        true_values = np.asarray(true_values)
        true_values = true_values.reshape(-1, 1) # So you can subtract it...
        debiased_estimates = all_estimates - self.biases

        print("get the shapes you dignus")
        import pdb; pdb.set_trace()

        covariance = debiased_estimates.T @ debiased_estimates
        covariance = covariance / num_samples
        print(f"Shape of covariance is {covariance.shape}")

        assert covariance.shape == (len(all_estimates[0]), len(all_estimates[0]))
        return covariance


        # debiased_estimates = np.asarray(debiased_estimates)
        # difference_matrix = debiased_estimates - true_values # The mean should be very close to zero...
        # assert np.allclose(difference_matrix.mean(axis=0), 0) # Make sure we're using debiased..
        # difference_matrix = all_estimates - true_values

        # covariance = np.cov(difference_matrix.T) # Not sure why we need the transpose, but we do.
        # print(f"Shape of covariance is {covariance.shape}")

        # assert covariance.shape == (len(all_estimates[0]), len(all_estimates[0]))
        # return covariance
        # independent_variance = np.var(difference_matrix, axis=0)

        # print(f"shape of independent_variance is {independent_variance.shape}")
        # return independent_variance

    def _calculate_weight_from_covariance(self, cov):
        # Assumes we already "debiased" using running average.
        # Eq. 2.2 of https://projecteuclid.org/download/pdf_1/euclid.lnms/1196285392
        assert len(cov.shape) == 2
        assert cov.shape[0] == cov.shape[1] == self.num_rollouts

        ones = np.ones((1,self.num_rollouts,), dtype=np.float32)
        inv_cov = np.linalg.inv(cov)

        w_prime = ones @ inv_cov / (ones @ inv_cov @ ones.T)

        return w_prime

    def _update_weights(self, experiences):
        """Updates bias, weight, and covariance"""

        # This should use global bias.
        cprint("Currently this is wrong, because it doesn't do the bias correctly.", "red")
        gamma = self.gamma


        num_rollouts = self.num_rollouts

        states = [exp.state for exp in experiences]
        actions = [exp.action for exp in experiences]

        true_values = self._get_true_state_values(experiences, gamma=gamma)
        all_estimates = self._get_values_for_rollouts(states)
        # all_estimates = self.create_rollout_matrix_from_data(
        #     states, actions, num_rollouts, true_values, gamma=gamma)

        # import ipdb; ipdb.set_trace()

        bias_est = self._calculate_bias_for_rollouts(all_estimates, true_values)
        self.biases = self.mixing_speed * self.biases + (1 - self.mixing_speed) * bias_est

        covariance_est = self._calculate_covariance_for_rollouts(all_estimates, true_values)
        self.covariances = self.mixing_speed * self.covariances + (1 - self.mixing_speed) * covariance_est

        self.weights = self._calculate_weight_from_covariance(self.covariances)


        # debiased_estimates = all_estimates - self.biases





        # bias, covariance = self._create_bias_covariance_from_data(experiences, gamma=gamma)
        # weights = self._calculate_weight_from_covariance(covariance)

        # assert np.all(np.greater_equal(weights, 0.0))


        # self.biases = ((1 - self.mixing_speed) * self.biases) + self.mixing_speed * bias
        # self.weights = ((1 - self.mixing_speed) * self.weights) + self.mixing_speed * weights
 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--seed", type=int, help="Random seed for this run (default=0)", default=0)
    parser.add_argument("--episodes", type=int, help="# episodes", default=NUM_EPISODES)
    parser.add_argument("--steps", type=int, help="# steps", default=NUM_STEPS)
    parser.add_argument("--render", type=bool, help="Render the mdp env", default=False)
    parser.add_argument("--pixel_observation", action='store_true', help="Images / Dense input", default=False)
    parser.add_argument("--exploration_method", type=str, default="eps-greedy")
    parser.add_argument("--eval_eps", type=float, default=0.05)
    parser.add_argument("--tensor_log", type=bool, default=False)
    parser.add_argument("--env", type=str, default="Acrobot-v1")
    args = parser.parse_args()

    logdir = create_log_dir(args.experiment_name)
    learning_rate = 1e-3 # 0.00025 for pong

    overall_mdp = GymMDP(env_name=args.env, pixel_observation=args.pixel_observation, render=args.render,
                         clip_rewards=False, term_func=None, seed=args.seed)
    ### THIS ONE WORKS FINE SO LONG AS YOU HAVE PIXEL OBSERVATIONS ####
    # overall_mdp = GymMDP(env_name="MontezumaRevengeNoFrameskip-v0", pixel_observation=args.pixel_observation, render=args.render,
    #                      clip_rewards=False, term_func=None, seed=args.seed)
    ### END ###
    # overall_mdp = GymMDP(env_name="MontezumaRevengeNoFrameskip-v4", pixel_observation=args.pixel_observation, render=args.render,
    #                      clip_rewards=False, term_func=None, seed=args.seed)
    # overall_mdp = GymMDP(env_name="CartPole-v0", pixel_observation=args.pixel_observation, render=args.render,
    #                         clip_rewards=False, term_func=None, seed=args.seed)

    # overall_mdp = LunarLanderMDP(render=args.render, seed=args.seed)

    state_dim = overall_mdp.env.observation_space.shape if args.pixel_observation else overall_mdp.env.observation_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    ddqn_agent = DQNAgent(state_size=state_dim, action_size=len(overall_mdp.actions),
                          seed=args.seed, device=device,
                          name="GlobalDDQN", lr=learning_rate, tensor_log=args.tensor_log, use_double_dqn=True,
                          exploration_method=args.exploration_method, pixel_observation=args.pixel_observation,
                          evaluation_epsilon=args.eval_eps)
    ddqn_episode_scores, s_ri_buffer = train(ddqn_agent, overall_mdp, args.episodes, args.steps)
    save_all_scores(args.experiment_name, logdir, args.seed, ddqn_episode_scores)
