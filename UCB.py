import numpy as np
import math

class UCB1(object):
    def __init__(self, n_arms):
        self.n = n_arms
        self.counts = [0] * n_arms
        self.values = [0] * n_arms
        self.rewards = 0

    def choose_arm(self):
        return np.argmax(self.get_values())

    def update(self, arm, reward):
        self.rewards = self.rewards + reward
        self.counts[arm] = self.counts[arm] + 1
        n = self.counts[arm]
        value = self.values[arm]
        new_value = value * (n - 1) / n + 1 * reward / n
        self.values[arm] = new_value

    def get_values(self):
        values = [0] * self.n

        for (i, v) in enumerate(values):
            if self.counts[i] == 0:
                values[i] = float("inf")
            else:
                values[i] = self.values[i] + math.sqrt(2 * math.log(self.n) / self.counts[i])
        
        return values
