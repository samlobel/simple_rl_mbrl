"""
Alright, what's the big difference between this one and the other main?
Well, instead of having something that learns a covariance matrix separately,
this is much more end to end. It learns covariance on-policy, and then
from there uses it to make better updates to the policy.

We could do this with an actor-critic type architecture to be sure.
But don't forget that we can definitely also just do this with softmax-sampling.
If we take 10 softmax-samples, we can get an average value.

So the new pieces we have are, something that takes in a Q-function and a
world-model, keeps track of covariance, and then uses rollouts and sampling to
make more accurate targets given data. I like it.
"""