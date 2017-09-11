from matplotlib import pyplot as plt
import json
import numpy as np

rewards = []
rew_means = []
ep_len = 0
with open('acktr/logs/openai-2017-09-04-18-09-26-193922/monitor.json') as file:
    for line in file:
        j = json.loads(line)
        if "r" in j:
            rew = j["r"]
            if np.isnan(rew):
                break
            else:
                rewards.append(rew)
        if "l" in j:
            ep_len = j["l"]
print(ep_len)
print(len(rewards))
#rewards = np.clip(rewards, -40,0)
slice_len = int(len(rewards) / 12.5)
for i in range(slice_len):
    rew_means.append(np.mean(rewards[i:i+slice_len]))

plt.plot(rew_means)
plt.show()


"""
rewards = []

with open('pposgd/logs/openai-2017-09-05-01-27-21-513543/progress.json') as file:
    for line in file:
        j = json.loads(line)
        rew = j["EpRewMean"]
        if np.isnan(rew):
            break
        else:
            rewards.append(rew)

rewards = np.clip(rewards, -40,0)
plt.plot(rewards)
plt.show()
"""
