import numpy as np
import math

class EpsilonGreedy(object):
    def __init__(self, n_arms, c = 20, d = 0.05):
        self.n = n_arms
        self.counts=[0] * n_arms
        self.values = [0] * n_arms
        self.rewards = 0
        """ c and d control exploration and exploition rate"""
        self.c = c
        self.d = d

    def choose_arm(self):
        epsilon = self.get_epsilon()
        if np.random.random() > epsilon:
            return np.argmax(self.values)
        else:
            return np.random.randint(self.n)

    def update(self, arm, reward):
        self.rewards = self.rewards + reward
        self.counts[arm] = self.counts[arm] + 1
        n = self.counts[arm]
        value = self.values[arm]
        new_value = ((n-1)/float(n)) * value + (1/float(n)) * reward
        self.values[arm] = new_value

    def get_epsilon(self):
        total = np.sum(self.counts)
        if total == 0:
            return 1
        else:
            return np.min([1, float(self.c * self.n) / (total * math.pow(self.d, 2))])

