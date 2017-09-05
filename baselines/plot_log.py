from matplotlib import pyplot as plt
import json
import numpy as np

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
