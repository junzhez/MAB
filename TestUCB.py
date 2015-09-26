from UCB import *

arms = [0.5, 0.6, 0.7]
d = np.max([ v for v in (np.max(arms) - arms) if v != 0])

player = UCB1(np.size(arms))

rounds = range(0, 100000)

for t in rounds:
    i = player.choose_arm()
    reward = np.random.binomial(1, arms[i])
    player.update(i, reward)

print(player.values)
print(player.rewards)
