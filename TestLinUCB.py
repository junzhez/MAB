from LinUCB import *

arms = [[3, 4, 5], [2, 3, 4], [1, 2, 3]]

n_arms = len(arms)
dim = len(arms[0])

player = LinUCB(n_arms, dim)

rounds = range(0, 10000)

rewards = 0

for t in rounds:
    x = np.matrix(np.random.randint(2, size=dim)).transpose()
    i = player.choose_arm(x)
    e = np.random.randn()
    reward = np.matrix(arms[i]) * x + e
    player.update(x, i, reward)

    rewards = rewards + np.max(arms * x + e)

print((rewards - player.rewards) / rounds.stop)
print(player.theta)
