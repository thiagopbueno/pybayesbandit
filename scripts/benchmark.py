#! /usr/bin/env python3

# This file is part of pybayesbandit.

# pybayesbandit is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# pybayesbandit is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with pybayesbandit. If not, see <http://www.gnu.org/licenses/>.


from pybayesbandit.bandits.bernoulli import BernoulliBandit
from pybayesbandit.learners.random import RandomPolicy
from pybayesbandit.learners.ucb import UCBPolicy
from pybayesbandit.learners.thompson import ThompsonSamplingPolicy
from pybayesbandit.learners.vi import BetaBernoulliVIPolicy
from pybayesbandit.learners.uct import BetaBernoulliUCTPolicy
from pybayesbandit.game import Game

import argparse
import matplotlib.pyplot as plt
import numpy as np
import time


policies = {
    'Random': RandomPolicy,
    'UCB': UCBPolicy,
    'TS': ThompsonSamplingPolicy,
    'VI': BetaBernoulliVIPolicy,
    'UCT': BetaBernoulliUCTPolicy
}

probs=[0.42, 0.35, 0.81]
bandit = BernoulliBandit(probs)

N = 200
horizons = range(10, 41, 5)

results = {}
learners = {}
for name, policy in policies.items():
    print('\n{} is playing ...'.format(name))

    avg_regrets, std_regrets, avg_rewards, std_rewards = [], [], [], []
    learners[name] = []

    for T in horizons:
        start = time.time()

        learner = policy(bandit.size, T)
        learners[name].append(learner)

        print('T = {:4d}: '.format(T), end='')
        game = Game(bandit, learner)
        avg_reward, std_reward, avg_regret, std_regret = game.run(N, T)

        end = time.time()
        uptime = end - start
        print('done in {:.4f} sec.'.format(uptime))

        avg_regrets.append(avg_regret)
        std_regrets.append(std_regret)
        avg_rewards.append(avg_reward)
        std_rewards.append(std_reward)

    results[name] = (
        np.array(avg_regrets), np.array(std_regrets),
        np.array(avg_rewards), np.array(std_rewards)
    )

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

for name, (avg_regrets, std_regrets, avg_rewards, std_rewards) in results.items():
    ax1.plot(horizons, avg_regrets, label=name)
    ax1.fill_between(horizons, avg_regrets - std_regrets, avg_regrets + std_regrets, alpha=0.2)
    ax2.plot(horizons, avg_rewards, label=name)
    ax2.fill_between(horizons, avg_rewards - std_rewards, avg_rewards + std_rewards, alpha=0.2)

ax1.set_title('Regret', fontweight='bold')
ax1.set_xlabel('Horizon')
ax1.set_xticks(horizons)
ax1.legend()
ax1.grid()

ax2.set_title('Reward', fontweight='bold')
ax2.set_xlabel('Horizon')
ax2.set_xticks(horizons)
ax2.legend()
ax2.grid()

plt.show()
