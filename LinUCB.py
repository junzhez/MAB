import numpy as np
import math

class LinUCB(object):
    def __init__(self, n_arms, dim, a = 2):
        self.n = n_arms
        self.A = [np.matrix(np.identity(dim))] * n_arms
        self.theta = [[]] * n_arms
        self.b = [np.matrix(np.zeros([dim, 1]))] * n_arms
        self.a = a
        self.rewards = 0

    def choose_arm(self, x):
        probs = [0] * self.n
        for i, m in enumerate(self.A):
            self.theta[i] = np.linalg.inv(m) * self.b[i]
            probs[i] = self.theta[i].transpose() * x + self.a * np.math.sqrt(x.transpose() * np.linalg.inv(m) * x)
        
        return np.argmax(probs)

    def update(self, x, arm, reward):
        self.rewards = self.rewards + reward
        new_A = self.A[arm] + x * x.transpose()
        self.A[arm] = new_A

        new_b = self.b[arm] + x * reward
        self.b[arm] = new_b
