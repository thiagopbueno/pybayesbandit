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

from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import time
import sys

UCTParams = namedtuple('UCTParams', 'trials maxdepth')

policies = {
    # 'Random': RandomPolicy,
    'UCB1': UCBPolicy,
    'TS': ThompsonSamplingPolicy,
    # 'VI': BetaBernoulliVIPolicy,
    'UCT(T=15, D=5)': (BetaBernoulliUCTPolicy, UCTParams(trials=15, maxdepth=5)),
    'UCT(T=15, D=15)': (BetaBernoulliUCTPolicy, UCTParams(trials=15, maxdepth=15)),
    'UCT(T=15, D=30)': (BetaBernoulliUCTPolicy, UCTParams(trials=15, maxdepth=30)),
}

N = 10
T = 1000

Ks = [2, 10, 30]
deltas = [0.05, 0.25, 0.40]

fig = plt.figure(figsize=(20, 10))

i = 1
for delta in deltas:

    for K in Ks:

        probs = [0.5] * (K - 1) + [0.5 + delta]
        bandit = BernoulliBandit(probs)

        print('\n>> Bernoulli(K={}, delta={}) ...'.format(K, delta))

        results = {}

        for name, policy in policies.items():
            print('\n{} is playing ... '.format(name), end='')

            start = time.time()

            if isinstance(policy, tuple):
                learner = policy[0](bandit.size, T, policy[1])
            else:
                learner = policy(bandit.size, T)

            game = Game(bandit, learner)
            results[name] = game.run(N, T)

            end = time.time()
            uptime = end - start
            print('done in {:.4f} sec.'.format(uptime))

        fig.add_subplot(len(deltas), len(Ks), i)

        rounds = range(1, T+1)
        for name, (rewards, regrets) in results.items():
            plt.plot(rounds, regrets[0], label=name)
            # plt.fill_between(rounds, regrets[0] - regrets[1], regrets[0] + regrets[1], alpha=0.2)

        plt.title('Regret (K={}, delta={})'.format(K, delta), fontweight='bold')
        plt.xlabel('rounds (t)')
        plt.xticks(np.arange(1, 11, dtype=np.int32) * int(T / 10))
        plt.legend()
        plt.grid()
        plt.tight_layout()

        i += 1


filename = sys.argv[1]
plt.savefig('{}.pdf'.format(filename), format='pdf')
