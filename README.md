# SAM NOTES:

WHERE DID I LEAVE OFF? I wrote something to get me a state-independent bias/variance for each rollout. But I still need to do the thing that gets it the data in the first place.

SHOOT. The bias part is pretty unnecessary as far as actually adding the values. But it is helpful as far as calculating the variance per sample goes. I don't know actually.
Actually, honestly it's not so clear how the bias comes into play. We get the difference between the real and the calculated for each depth. That's the thing that we're measuring.

There's still something better about it though, because STEVE doesn't directly compare to the truth, only to its own uncertainty. So, if STEVE is pretty sure about the values for the first estimate, it might assign a high value to it. Regardless of whether the first one is historically close to right. But mine would realize, the first one ain't all that actually.

So, we could see how we would potentially adjust things in the case that our Q-function was noisy and biased. I'm really not sure. I guess another weird thing is, STEVE uses variance within a set to determine if you should trust it. BUT, isn't that sort of ignoring the fact that the mean value for all the different lengths is going to be different? That should probably tell you something in itself. But I guess that's the point, is you have a bunch of different measurements, and you get their variance, and then you can take their weighted average? Seems very smart to me. But you can definitely imagine having a limited function class and ending up with the same answer for all of a certain state's values, but them all being wrong in the same direction... Not sure exactly how this should work out.

I think we could probably do an example with a linear model, that is helpful despite being not great. Because probably one of the bigger sources of bias or correlated variance is a too-weak model.


To run Akhil's stuff:
`python DQNAgentClass.py --experiment_name testing-exp --pixel_observation True`
But I'm going to switch it to working without videos, because no way I'm doing this on videos.
I want it to work with: 
`python main.py --experiment_name testing-exp --env CartPole-v1`


MAYBE I should be just trying to get it to work with one-step models. It's not perfect,
but it's the way people have done it for a long time now. And it means I can use other people's replay buffers and whatnot. I think I'll do that, it's not perfect but it's something.

But, we do need some sort of long transition in order to generate a target for the correlation matrix or whatever.

For now, I can do that as an outer loop of some sort? I mean, we're working with pretty fast environments with small state spaces...

So, I need a training function. It should be doing something like training a DQN alongside the models. This really shouldn't be all that hard. It would be nice if we could somehow integrate these things all to be part of one object. That way, among other things, we could make our learner really "on policy" instead of just acting with accordance to the DQN. Would we need target-networks for the models? I don't think so, since there's no chance of explosion or whatever.


Small complication: if we make our covariance matrix on the fly, how do we deal with off-policy anything?
One option is to include epsilon greedy in rollouts. Another is to only train it on-policy after first
action... Not sure really.

There's a bunch of stuff about steps because it's the options framework. I don't really need any of that, should I just completely take it out? No, it's set to 1 here, that' really fine.

can't keep up with logging it seems, because it's logging too fast. making it not log every step means it's doing a lot better...

Okay, so: what do I want to do exactly? I think I want to act totally on-policy, at least to start.
I'll need a calculate-covariance function. I'll need a "calculate_bias_covariance_from_data function."
I need something that actually generates the data first. It'll end up being something simple like a matrix that has a bunch of values, that we then fit a linear model to? Should we allow for that? I don't really know to be honest.

I can first do a simple comparison that implements TD-lambda, and compares against how it does without it. How do I compare? I think I need to make a new function, that runs a test occasionally, and maybe logs the results? Maybe average over 100 trials or something.






# simple_rl
A simple framework for experimenting with Reinforcement Learning in Python.

There are loads of other great libraries out there for RL. The aim of this one is twofold:

1. Simplicity.
2. Reproducibility of results.

A brief tutorial for a slightly earlier version is available [here](http://cs.brown.edu/~dabel/blog/posts/simple_rl.html). As of version 0.77, the library should work with both Python 2 and Python 3. Please let me know if you find that is not the case!

simple_rl requires [numpy](http://www.numpy.org/) and [matplotlib](http://matplotlib.org/). Some MDPs have visuals, too, which requires [pygame](http://www.pygame.org/news). Also includes support for hooking into any of the [Open AI Gym environments](https://gym.openai.com/envs). I recently added a basic test script, contained in the _tests_ directory.


## Installation

The easiest way to install is with [pip](https://pypi.python.org/pypi/pip). Just run:

	pip install simple_rl

Alternatively, you can download simple_rl [here](https://github.com/david-abel/simple_rl/tarball/v0.76).

## Example

Some examples showcasing basic functionality are included in the [examples](https://github.com/david-abel/simple_rl/tree/master/examples) directory.

To run a simple experiment, import the _run_agents_on_mdp(agent_list, mdp)_ method from _simple_rl.run_experiments_ and call it with some agents for a given MDP. For example:

	# Imports
	from simple_rl.run_experiments import run_agents_on_mdp
	from simple_rl.tasks import GridWorldMDP
	from simple_rl.agents import QLearningAgent

	# Run Experiment
	mdp = GridWorldMDP()
	agent = QLearningAgent(mdp.get_actions())
	run_agents_on_mdp([agent], mdp)

Running the above code will run unleash _Q_-learning on a simple GridWorld. When it finishes it will store the results in _cur_dir/results/*_ and open the following plot:

<img src="https://david-abel.github.io/blog/posts/images/simple_grid.jpg" width="480" align="center">

For a slightly more complicated example, take a look at the code of _simple_example.py_. Here we run three few agents on the grid world from the Russell-Norvig AI textbook:

	from simple_rl.agents import QLearningAgent, RandomAgent, RMaxAgent
	from simple_rl.tasks import GridWorldMDP
	from simple_rl.run_experiments import run_agents_on_mdp

    # Setup MDP.
    mdp = GridWorldMDP(width=4, height=3, init_loc=(1, 1), goal_locs=[(4, 3)], lava_locs=[(4, 2)], gamma=0.95, walls=[(2, 2)])

    # Setup Agents.
    ql_agent = QLearningAgent(actions=mdp.get_actions())
    rmax_agent = RMaxAgent(actions=mdp.get_actions())
    rand_agent = RandomAgent(actions=mdp.get_actions())

    # Run experiment and make plot.
    run_agents_on_mdp([ql_agent, rmax_agent, rand_agent], mdp, instances=5, episodes=50, steps=10)

The above code will generate the following plot:

<img src="https://david-abel.github.io/blog/posts/images/rn_grid.jpg" width="480" align="center">

## Overview

* (_agents_): Code for some basic agents (a random actor, _Q_-learning, [[R-Max]](http://www.jmlr.org/papers/volume3/brafman02a/brafman02a.pdf), _Q_-learning with a Linear Approximator, and so on).

* (_experiments_): Code for an Experiment class to track parameters and reproduce results.

* (_mdp_): Code for a basic MDP and MDPState class, and an MDPDistribution class (for  lifelong learning). Also contains OO-MDP implementation [[Diuk et al. 2008]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.149.7056&rep=rep1&type=pdf).

* (_planning_): Implementations for planning algorithms, includes ValueIteration and MCTS [[Couloum 2006]](https://hal.archives-ouvertes.fr/file/index/docid/116992/filename/CG2006.pdf), the latter being still in development.

* (_tasks_): Implementations for a few standard MDPs (grid world, N-chain, Taxi [[Dietterich 2000]](http://www.scs.cmu.edu/afs/cs/project/jair/pub/volume13/dietterich00a.pdf), and the [OpenAI Gym](https://gym.openai.com/envs)).

* (_utils_): Code for charting and other utilities.

## Contributing

If you'd like to contribute: that's great! Take a look at some of the needed improvements below: I'd love for folks to work on those pieces. Please see the [contribution guidelines](https://github.com/david-abel/simple_rl/blob/master/CONTRIBUTING.md). Email me with any questions.

## Making a New MDP

Make an MDP subclass, which needs:

* A static variable, _ACTIONS_, which is a list of strings denoting each action.

* Implement a reward and transition function and pass them to MDP constructor (along with _ACTIONS_).

* I also suggest overwriting the "\_\_str\_\_" method of the class, and adding a "\_\_init\_\_.py" file to the directory.

* Create a State subclass for your MDP (if necessary). I suggest overwriting the "\_\_hash\_\_", "\_\_eq\_\_", and "\_\_str\_\_" for the class to play along well with the agents.


## Making a New Agent

Make an Agent subclass, which requires:

* A method, _act(self, state, reward)_, that returns an action.

* A method, _reset()_, that puts the agent back to its _tabula rasa_ state.

## In Development

I'm hoping to add the following features:

* __Planning__: Finish MCTS [[Coloum 2006]](https://hal.inria.fr/file/index/docid/116992/filename/CG2006.pdf), implement RTDP [[Barto et al. 1995]](https://pdfs.semanticscholar.org/2838/e01572bf53805c502ec31e3e00a8e1e0afcf.pdf)
* __Deep RL__: Write a DQN [[Mnih et al. 2015]](http://www.davidqiu.com:8888/research/nature14236.pdf) in PyTorch, possibly others (some kind of policy gradient).
* __Efficiency__: Convert most defaultdict/dict uses to numpy.
* __Docs__: Tutorials, contribution policy, and thorough documentation.
* __Visuals__: Unify MDP visualization.
* __Misc__: Additional testing, reproducibility checks (store more in params file, rerun experiment from params file).

Cheers,

-Dave