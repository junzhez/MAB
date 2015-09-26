import numpy as np
import math

class Exp3(object):
    def __init__(self, n_arms, r):
        self.n = n_arms
        self.weights = [1] * n_arms
        self.probs = [float(1/n_arms)] * n_arms
        self.rewards = 0
        self.r = r

    def choose_arm(self):
        return np.argmax(np.random.multinomial(1, self.probs))

    def update(self, arm, reward):
        self.rewards = self.rewards + reward
        prob = self.probs[arm]
        values = [0] * self.n
        values[arm] = reward/prob
        new_weights = [v * math.exp(self.r * values[i] / self.n) for (i, v) in enumerate(self.weights)]
        self.weights = new_weights
        print(np.sum(self.weights))
        new_probs = self.weights / np.sum(self.weights)
        self.probs = [v * (1 - self.r) + self.r / self.n for v in new_probs]

