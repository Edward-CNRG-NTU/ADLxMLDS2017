import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

step_history = []
reward_history = []
zeros_history = []
loss_history = []

prj_dir = 'agent_dir/'
# prj_dir = 'agent_dir/gcp/gcp_4_dqn_basic_done_gamma_0.75/'
# prj_dir = 'agent_dir_pg_3/log'
with open(prj_dir + 'log', 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break

        segments = line.split(' ')
        try:
            step = int(segments[2])
            reward = int(segments[4])
            zeros = float(segments[6])
            if len(segments) > 10:
                loss = float(segments[10])
            else:
                loss = 0.
        except IndexError:
            print(segments)
            continue

        step_history.append(step)
        reward_history.append(reward)
        zeros_history.append(zeros)
        loss_history.append(loss)

print(len(step_history), sum(step_history))
plt.figure('step_history')
plt.plot(np.array(step_history))
plt.figure('reward_history')
plt.plot(np.array(reward_history))
N = 30
plt.plot(np.convolve(np.array(reward_history), np.ones((N,))/N, mode='same'), 'r')
N = 300
plt.plot(np.convolve(np.array(reward_history), np.ones((N,))/N, mode='same'), 'g')
plt.savefig(filename=prj_dir + 'reward_history.png')
plt.figure('zeros_ratio')
plt.plot(np.array(zeros_history))
plt.figure('loss_history')
plt.plot(np.array(loss_history))

plt.show()
