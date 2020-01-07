import numpy as np
import random
from collections import namedtuple, deque, defaultdict
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
from simple_rl.agents.func_approx.sam_stuff.DQNAgentClass import WorldModel
from simple_rl.agents.func_approx.sam_stuff.ModelQNetworkComposer import Composer


NUM_EPISODES = 3500
NUM_STEPS = 10000



# def test_forward_pass(dqn_agent, mdp):
#     # load the weights from file
#     mdp.reset()
#     state = deepcopy(mdp.init_state)
#     overall_reward = 0.
#     mdp.render = True

#     while not state.is_terminal():
#         action = dqn_agent.act(state.features(), train_mode=False)
#         reward, next_state = mdp.execute_agent_action(action)
#         overall_reward += reward
#         state = next_state

#     mdp.render = False
#     return overall_reward


def show_video(dqn_agent, mdp):
    # load the weights from file
    mdp.reset()
    state = deepcopy(mdp.init_state)
    overall_reward = 0.
    mdp.render = True

    while not state.is_terminal():
        action = dqn_agent.act(state.features(), train_mode=False)
        reward, next_state = mdp.execute_agent_action(action)
        overall_reward += reward
        state = next_state

    mdp.render = False
    return overall_reward


def save_all_scores(experiment_name, log_dir, seed, scores):
    print("\rSaving training and validation scores..")
    training_scores_file_name = "{}_{}_training_scores.pkl".format(experiment_name, seed)

    if log_dir:
        training_scores_file_name = os.path.join(log_dir, training_scores_file_name)

    with open(training_scores_file_name, "wb+") as _f:
        pickle.dump(scores, _f)

def create_log_dir(experiment_name):
    path = os.path.join(os.getcwd(), experiment_name)
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    return path

def test_render(agent, mdp):
    while True:
        print("Press ctrl-C to quit")
        mdp.set_render(True)
        mdp.reset()
        state = mdp.init_state
        while True:
            action = agent.act(state.features(), train_mode=False)
            reward, next_state = mdp.execute_agent_action(action)
            state = next_state

            game_over = mdp.game_over if hasattr(mdp, 'game_over') else False
            if state.is_terminal() or game_over:
                print('bye bye')
                break


def collect_data_for_bias_variance_calculation(mdp, q_agent, num_runs):
    """
    Runs on-policy, and just makes the data that we'll pass to the composer.
    """
    exp = namedtuple("Experience", field_names=["state","action","reward","next_state", "done", "time_limit_truncated"])
    experiences = []

    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    time_limit_truncateds = []
 
    for _ in range(num_runs):
        mdp.reset()
        state = deepcopy(mdp.init_state)
        state = np.asarray(state.features())

        true_finish = False
        while True:
            # action = agent.act(state.features(), train_mode=True)
            # reward, next_state = mdp.execute_agent_action(action)


            action = composer.q_agent.get_best_action(state)
            reward, next_state = mdp.execute_agent_action(action)
            # is_terminal = next_state.is_terminal()
            # time_limit_truncated = next_state.is_time_limit_truncated()


            experiences.append(
                exp(state=state,
                    action=action,
                    reward=reward,
                    next_state=np.asarray(next_state.features()),
                    done=next_state.is_terminal(),
                    time_limit_truncated=next_state.is_time_limit_truncated()
                    ))

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.asarray(next_state.features()))
            dones.append(next_state.is_terminal())
            time_limit_truncateds.append(next_state.is_time_limit_truncated())

            game_over = mdp.game_over if hasattr(mdp, 'game_over') else False

            if game_over and not next_state.is_terminal():
                print('howza')
                # import ipdb; ipdb.set_trace()
                raise Exception("Honestly, we're just not dealing with this well here.")

            if next_state.is_terminal():
                break

            state = np.asarray(next_state.features())


    return experiences

    # return dict(
    #     states=states,
    #     actions=actions,
    #     rewards=rewards,
    #     next_states=next_states,
    #     dones=dones,
    #     time_limit_truncateds=time_limit_truncateds,
    # )
            


    pass


class Evaluator:

    def __init__(self, mdp, composer, num_runs_each=1, rollout_depth=5, lambdas_to_test=None, logdir="."):
        self.mdp = mdp
        self.composer = composer
        self.num_runs_each = num_runs_each
        self.rollout_depth = rollout_depth
        self.logdir = logdir

        self._bias = None
        self._variance = None

        if lambdas_to_test is None:
            self.lambdas_to_test = [0.0, 0.5, 1.0]
        else:
            self.lambdas_to_test = lambdas_to_test


        self.results = defaultdict(list)

    def _set_bias_variance(self, num_runs_to_collect_over):
        data = collect_data_for_bias_variance_calculation(self.mdp, self.composer.q_agent, num_runs_to_collect_over)
        bias, variance = self.composer.create_bias_variance_from_data(data, 5)
        self._bias = bias
        self._variance = variance
        print(f"Bias: {bias}\nVariance: {variance}")

    def evaluate_different_models(self, *, training_steps):
        """
        This does the evaluation, prints out results, but then importantly
        populates some storage list, which we can then use to make plots.
        """
        assert self._bias is not None
        assert self._variance is not None

        lambdas_to_test = self.lambdas_to_test
        mdp = self.mdp
        composer = self.composer
        num_runs_each = self.num_runs_each
        rollout_depth = self.rollout_depth

        funcs = [(lam, lambda s: composer.get_best_action_td_lambda(s, rollout_depth, gamma=0.99, lam=lam))
                for lam in lambdas_to_test]

        funcs.append(("OptimalVariance",
            lambda s: composer.get_best_action_for_bias_variance(s, rollout_depth, self._bias, self._variance, gamma=0.99)))

        # for lam in lambdas_to_test:
        for key, func in funcs:
            all_rewards = []
            for _ in range(num_runs_each):
                mdp.reset()
                state = deepcopy(mdp.init_state)
                state = np.asarray(state.features())
                reward_so_far = 0.0
                while True:
                    # state = torch.from_numpy(state).float().unsqueeze(0).to("cuda")
                    # action = composer.get_best_action_td_lambda(state, rollout_depth, gamma=0.99, lam=lam)
                    action = func(state)
                    reward, next_state = mdp.execute_agent_action(action)
                    reward_so_far += reward
                    game_over = mdp.game_over if hasattr(mdp, 'game_over') else False
                    if next_state.is_terminal() or game_over:
                        break

                    state = np.asarray(next_state.features())
                self.results[key].append((training_steps, reward_so_far))
                all_rewards.append(reward_so_far)
            all_rewards = np.asarray(all_rewards)
            print(f"{num_runs_each} runs:     Key={key}, Reward={np.mean(all_rewards)} ({np.std(all_rewards)})")
            print(all_rewards)

    def write_graphs(self):
        plt.figure()
        for lam, vals in self.results.items():
            xs, ys = zip(*vals)
            ax = sns.lineplot(x=xs, y=ys, label=f"Lam={lam}")

        plt.savefig(os.path.join(self.logdir, "results.png"))
        # plt.show()
        plt.clf()


def evaluate_different_models(mdp, composer, num_runs_each=1, training_steps=None):
    # Somehow I want to also graph this... How should I do that?
    # I could make this a class, and keep track of past things. But that does
    # seem heavy-handed. How about I start by just printing them out...
    lambdas_to_test = [0.0, 0.5, 1.0]
    rollout_depth = 5

    for lam in lambdas_to_test:
        all_rewards = []
        for _ in range(num_runs_each):
            mdp.reset()
            state = deepcopy(mdp.init_state)
            state = np.asarray(state.features())
            reward_so_far = 0.0
            while True:
                # state = torch.from_numpy(state).float().unsqueeze(0).to("cuda")
                action = composer.get_best_action_td_lambda(state, rollout_depth, gamma=0.99, lam=lam)
                reward, next_state = mdp.execute_agent_action(action)
                reward_so_far += reward
                game_over = mdp.game_over if hasattr(mdp, 'game_over') else False
                if next_state.is_terminal() or game_over:
                    break

                state = np.asarray(next_state.features())
            all_rewards.append(reward_so_far)
        all_rewards = np.asarray(all_rewards)
        print(f"{num_runs_each} runs:     Lam={lam}, Reward={np.mean(all_rewards)} ({np.std(all_rewards)})")
        print(all_rewards)


def train(agent, mdp, episodes, steps, init_episodes=10, *, save_every, logdir, world_model, composer):
    model_save_loc = os.path.join(logdir, 'model.tar')
    per_episode_scores = []
    last_10_scores = deque(maxlen=100)
    iteration_counter = 0
    state_ri_buffer = []

    # Observation and reward normalization
    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 84, 84))

    # Initialize the RMS normalizers
    if agent.exploration_method == "rnd":
        for episode in range(init_episodes):
            observation_buffer = []
            mdp.reset()
            init_observation = np.array(mdp.init_state.features())[-1, :, :]
            assert init_observation.shape == (84, 84), init_observation.shape
            observation_buffer.append(init_observation)
            while True:
                action = np.random.randint(0, overall_mdp.env.action_space.n)
                r, state = mdp.execute_agent_action(action)
                observation = np.array(state.features())[-1, :, :]
                observation_buffer.append(observation)
                if state.is_terminal():
                    break
            observation_batch = np.stack(observation_buffer)
            obs_rms.update(observation_batch)

    last_save = time.time()

    evaluator = Evaluator(mdp, composer, num_runs_each=5, rollout_depth=5, logdir=logdir)

    for episode in range(episodes):

        if episode % 10 == 0:
            print(f"Evaluating on episode {episode}")

            evaluator._set_bias_variance(10)
            evaluator.evaluate_different_models(training_steps=episode)
            print("At some point definitely make this a CL-Arg")
            evaluator.write_graphs()

        if time.time() - last_save > save_every:
            print("Saving Model")
            last_save = time.time()
            torch.save(agent.state_dict(), model_save_loc)

        mdp.reset()
        state = deepcopy(mdp.init_state)

        observation_buffer = []
        intrinsic_reward_buffer = []

        init_features = np.asarray(mdp.init_state.features())
        if len(init_features.shape) == 3:
            init_observation = init_features[-1, :, :]
            assert init_observation.shape == (84, 84), init_observation.shape
        else:
            init_observation = init_features

        #### FROM AKHIL
        # init_observation = np.array(mdp.init_state.features())[-1, :, :]
        # assert init_observation.shape == (84, 84), init_observation.shape
        observation_buffer.append(init_observation)

        score = 0.
        while True:
            iteration_counter += 1
            action = agent.act(state.features(), train_mode=True)
            reward, next_state = mdp.execute_agent_action(action)

            if agent.exploration_method == "rnd":
                observation = np.array(state.features())[-1, :, :]
                normalized_observation = ((observation - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)
                intrinsic_reward = agent.rnd.get_single_reward(normalized_observation)

                observation_buffer.append(observation)
                intrinsic_reward_buffer.append(intrinsic_reward)
                normalized_intrinsic_reward = (intrinsic_reward - reward_rms.mean) / np.sqrt(reward_rms.var)

                # Logging
                player_position = mdp.get_player_position()
                state_ri_buffer.append((player_position, normalized_intrinsic_reward))
                reward += normalized_intrinsic_reward

                if agent.tensor_log:
                    agent.writer.add_scalar("Normalized Ri", normalized_intrinsic_reward, iteration_counter)

            agent.step(state.features(), action, reward, next_state.features(), next_state.is_terminal(),
                num_steps=1, time_limit_truncated=next_state.is_time_limit_truncated())
            agent.update_epsilon()

            world_model.step(state.features(), action, reward, next_state.features(), next_state.is_terminal(),
                num_steps=1, time_limit_truncated=next_state.is_time_limit_truncated())

            state = next_state
            score += reward
            if agent.tensor_log:
                agent.writer.add_scalar("Score", score, iteration_counter)

            game_over = mdp.game_over if hasattr(mdp, 'game_over') else False
            if state.is_terminal() or game_over:
                break

        if agent.exploration_method == "rnd":
            reward_rms.update(np.stack(intrinsic_reward_buffer))
            obs_rms.update(np.stack(observation_buffer))

        last_10_scores.append(score)
        per_episode_scores.append(score)

        print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(episode, np.mean(last_10_scores), agent.epsilon), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(episode, np.mean(last_10_scores), agent.epsilon))
    return per_episode_scores, state_ri_buffer


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
    parser.add_argument("--tensor_log", default=False, action='store_true', help="Include this option if you want logging.")
    parser.add_argument("--env", type=str, default="Acrobot-v1")
    parser.add_argument("--save_every", type=int, help="Save every n seconds", default=60)
    parser.add_argument("--mode", type=str, help="'train' or 'view'", default='train')
    parser.add_argument("--epsilon_linear_decay", type=int, help="'train' or 'view'", default=100000)
    parser.add_argument("--use_softmax_target", default=False, action='store_true', help='When calculating backups, do you use the max or the softmax?')
    # parser.add_argument("--use_world_model", default=False, action='store_true', help="Include this option if you want to see how a world model trains.")
    args = parser.parse_args()

    logdir = create_log_dir(args.experiment_name)
    model_save_loc = os.path.join(logdir, 'model.tar')
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
    action_dim = len(overall_mdp.actions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    ddqn_agent = DQNAgent(state_size=state_dim, action_size=action_dim,
                        seed=args.seed, device=device,
                        name="GlobalDDQN", lr=learning_rate, tensor_log=args.tensor_log, use_double_dqn=True,
                        exploration_method=args.exploration_method, pixel_observation=args.pixel_observation,
                        evaluation_epsilon=args.eval_eps,
                        epsilon_linear_decay=args.epsilon_linear_decay,
                        use_softmax_target=args.use_softmax_target)

    world_model = WorldModel(state_size=state_dim, action_size=action_dim,
                        seed=args.seed, device=device,
                        name="WorldModel", lr=learning_rate, tensor_log=args.tensor_log,# use_double_dqn=True,
                        #exploration_method=args.exploration_method, pixel_observation=args.pixel_observation,
                        #evaluation_epsilon=args.eval_eps,
                        #epsilon_linear_decay=args.epsilon_linear_decay
                        )


    composer = Composer(
        q_agent=ddqn_agent,
        world_model=world_model,
        action_size=action_dim,
        device=device)

    data = collect_data_for_bias_variance_calculation(overall_mdp, ddqn_agent, 1)
    bias, variance = composer.create_bias_variance_from_data(data, 5)


    if args.mode == 'train':
        ddqn_episode_scores, s_ri_buffer = train(
            ddqn_agent, overall_mdp, args.episodes, args.steps, save_every=args.save_every, logdir=logdir, world_model=world_model,
            composer=composer)
        save_all_scores(args.experiment_name, logdir, args.seed, ddqn_episode_scores)
    elif args.mode == 'view':
        print('waow')
        print(model_save_loc)
        ddqn_agent.load_state_dict(torch.load(model_save_loc))
        test_render(ddqn_agent, overall_mdp)
        pass
    else:
        raise Exception("HEELLOOO")
