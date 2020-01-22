from collections import namedtuple, deque
import random
import torch
import numpy as np
import pdb


Transition = namedtuple('Transition', ("state", "action", "reward", "next_state", "done", "num_steps"))


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device, pixel_observation):
        """
        Initialize a ReplayBuffer object.
        Args:
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (torch.device): cpu / cuda:0 / cuda:1
            pixel_observation (bool): Whether observations are dense or images

        TODO: deque is very non-optimal for something that uses random.sample. Because
        it's O(n) to access. So, how about we just do a list with popping. Popping is bad as well.
        It needs to be cyclical... which is annoying for a bunch of reasons.
        Although, maybe we could do non-cyclical at first, and then once it fills up
        switch to cyclical! I like that.
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        # self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "num_steps"])
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "num_steps", "time_limit_truncated"])
        self.seed = random.seed(seed)
        np.random.seed(seed)
        self.device = device
        self.pixel_observation = pixel_observation

        self.positive_transitions = []

    def add(self, state, action, reward, next_state, done, num_steps, time_limit_truncated):
        """
        Add new experience to memory.
        Args:
            state (np.array): We add numpy arrays from gym env to the buffer, but sampling from buffer returns tensor
            action (int)
            reward (float_
            next_state (np.array)
            done (bool)
            num_steps (int): number of steps taken by the action/option to terminate
        """
        e = self.experience(state, action, reward, next_state, done, num_steps, time_limit_truncated)
        self.memory.append(e)

    def sample(self, batch_size=None):
        """Randomly sample a batch of experiences from memory."""
        size = self.batch_size if batch_size is None else batch_size
        experiences = random.sample(self.memory, k=size)

        # Log the number of times we see a non-negative reward (should be sparse)
        num_positive_transitions = sum([exp.reward >= 0 for exp in experiences])
        self.positive_transitions.append(num_positive_transitions)

        # With image observations, we need to add another dimension to the tensor before stacking
        if self.pixel_observation:
            states = torch.from_numpy(np.vstack([e.state[None, ...] for e in experiences if e is not None])).float().to(self.device)
        else:
            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        from termcolor import colored, cprint
        # cprint(f"Action shape: {actions.shape}", "red")
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        # cprint(f"Reward shape: {rewards.shape}", "red")
        if self.pixel_observation:
            next_states = torch.from_numpy(np.vstack([e.next_state[None, ...] for e in experiences if e is not None])).float().to(self.device)
        else:
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        steps = torch.from_numpy(np.vstack([e.num_steps for e in experiences if e is not None])).float().to(self.device)
        time_limit_truncateds = torch.from_numpy(np.vstack([e.time_limit_truncated for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        actions = actions.squeeze(1)
        rewards = rewards.squeeze(1)
        dones = dones.squeeze(1)
        time_limit_truncateds = time_limit_truncateds.squeeze(1)
        steps = steps.squeeze(1)

        return states, actions, rewards, next_states, dones, steps, time_limit_truncateds

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)




######
#TAKEN FROM STEVE (https://github.com/tensorflow/models/blob/master/research/steve/replay.py)
######

# from __future__ import print_function
# from future import standard_library
# standard_library.install_aliases()
# from builtins import zip
# from builtins import str
# from builtins import object
# # Copyright 2018 The TensorFlow Authors All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ==============================================================================

# import numpy as np
# import pickle
# import multiprocessing

# class ReplayBuffer(object):
#     """
#     Stores frames sampled from the environment, with the ability to sample a batch
#     for training.
#     """

#     def __init__(self, max_size, obs_dim, action_dim, roundrobin=True):
#         self.max_size = max_size
#         self.obs_dim = obs_dim
#         self.action_dim = action_dim
#         self.roundrobin = roundrobin

#         self.obs_buffer = np.zeros([max_size, obs_dim])
#         self.next_obs_buffer = np.zeros([max_size, obs_dim])
#         self.action_buffer = np.zeros([max_size, action_dim])
#         self.reward_buffer = np.zeros([max_size])
#         self.done_buffer = np.zeros([max_size])

#         self.count = 0

#     def random_batch(self, batch_size):
#         indices = np.random.randint(0, min(self.count, self.max_size), batch_size)

#         return (
#             self.obs_buffer[indices],
#             self.next_obs_buffer[indices],
#             self.action_buffer[indices],
#             self.reward_buffer[indices],
#             self.done_buffer[indices],
#             self.count
#         )

#     def add_replay(self, obs, next_obs, action, reward, done):
#         if self.count >= self.max_size:
#             if self.roundrobin: index = self.count % self.max_size
#             else:               index = np.random.randint(0, self.max_size)
#         else:
#             index = self.count

#         self.obs_buffer[index] = obs
#         self.next_obs_buffer[index] = next_obs
#         self.action_buffer[index] = action
#         self.reward_buffer[index] = reward
#         self.done_buffer[index] = done

#         self.count += 1

#     def save(self, path, name):
#         def _save(datas, fnames):
#             print("saving replay buffer...")
#             for data, fname in zip(datas, fnames):
#                 with open("%s.npz"%fname, "w") as f:
#                     pickle.dump(data, f)
#             with open("%s/%s.count" % (path,name), "w") as f:
#                 f.write(str(self.count))
#             print("...done saving.")

#         datas = [
#             self.obs_buffer,
#             self.next_obs_buffer,
#             self.action_buffer,
#             self.reward_buffer,
#             self.done_buffer
#         ]

#         fnames = [
#             "%s/%s.obs_buffer" % (path, name),
#             "%s/%s.next_obs_buffer" % (path, name),
#             "%s/%s.action_buffer" % (path, name),
#             "%s/%s.reward_buffer" % (path, name),
#             "%s/%s.done_buffer" % (path, name)
#          ]

#         proc = multiprocessing.Process(target=_save, args=(datas, fnames))
#         proc.start()

#     def load(self, path, name):
#         print("Loading %s replay buffer (may take a while...)" % name)
#         with open("%s/%s.obs_buffer.npz" % (path,name)) as f: self.obs_buffer = pickle.load(f)
#         with open("%s/%s.next_obs_buffer.npz" % (path,name)) as f: self.next_obs_buffer = pickle.load(f)
#         with open("%s/%s.action_buffer.npz" % (path,name)) as f: self.action_buffer = pickle.load(f)
#         with open("%s/%s.reward_buffer.npz" % (path,name)) as f: self.reward_buffer = pickle.load(f)
#         with open("%s/%s.done_buffer.npz" % (path,name)) as f: self.done_buffer = pickle.load(f)
#         with open("%s/%s.count" % (path,name), "r") as f: self.count = int(f.read())
