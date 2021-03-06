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
from pybayesbandit.learners.lookahead import LookaheadTreeSearchPolicy
from pybayesbandit.games.totalregret import TotalRegretGame
from pybayesbandit.games.simpleregret import SimpleRegretGame


from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import time
import sys


def plot_results(game_type, results, deltas, Ks, i, T):

    fig.add_subplot(len(deltas), len(Ks), i)

    if game_type == 'total':

        rounds = range(1, T+1)
        for name, (rewards, regrets) in results.items():
            plt.plot(rounds, regrets[0], label=name)
            # plt.fill_between(rounds, regrets[0] - regrets[1], regrets[0] + regrets[1], alpha=0.2)

        plt.title('Cumulative Regret ($K={}, \\Delta={}$)'.format(K, delta), fontweight='bold')
        plt.xlabel('rounds (t)')
        plt.xticks(np.arange(1, 11, dtype=np.int32) * int(T / 10))
        plt.legend()
        plt.grid()
        plt.tight_layout()

    elif game_type == 'simple':

        learners, regrets = list(results.keys()), list(results.values())
        avg_regret = [r[0] for r in regrets]
        std_regret = [r[1] for r in regrets]

        opacity = 0.4
        error_config = {'ecolor': '0.3'}

        plt.bar(learners, avg_regret, 0.35,
            alpha=opacity, color='b',
            yerr=std_regret, error_kw=error_config)

        plt.title('Simple Regret ($K={}, \\Delta={}$)'.format(K, delta), fontweight='bold')
        plt.xlabel('learner')
        plt.grid(axis='y')
        plt.tight_layout()



UCTParams = namedtuple('Params', 'trials maxdepth C')
AOTreeParams = namedtuple('Params', 'maxdepth')

policies = {
    # 'Random': RandomPolicy,
    'UCB1': UCBPolicy,
    'TS': ThompsonSamplingPolicy,
    # 'VI': BetaBernoulliVIPolicy,
    'AOTree(depth=1)': (LookaheadTreeSearchPolicy, AOTreeParams(maxdepth=1)),
    'AOTree(depth=2)': (LookaheadTreeSearchPolicy, AOTreeParams(maxdepth=2)),
    'AOTree(depth=3)': (LookaheadTreeSearchPolicy, AOTreeParams(maxdepth=3)),
    # 'AOTree(depth=4)': (LookaheadTreeSearchPolicy, AOTreeParams(maxdepth=4)),
    # 'AOTree(depth=5)': (LookaheadTreeSearchPolicy, AOTreeParams(maxdepth=5))
    # 'UCT(trials=64, depth=3, C=2)': (BetaBernoulliUCTPolicy, UCTParams(trials=64, maxdepth=3, C=2)),
    # 'UCT(trials=64, depth=3, C=5)': (BetaBernoulliUCTPolicy, UCTParams(trials=64, maxdepth=3, C=5)),
    'UCT(trials=64, depth=3, C=10)': (BetaBernoulliUCTPolicy, UCTParams(trials=64, maxdepth=3, C=10)),
    # 'UCT(trials=256, depth=4, C=2)': (BetaBernoulliUCTPolicy, UCTParams(trials=256, maxdepth=4, C=2)),
    # 'UCT(trials=256, depth=4, C=5)': (BetaBernoulliUCTPolicy, UCTParams(trials=256, maxdepth=4, C=5)),
    'UCT(trials=256, depth=4, C=10)': (BetaBernoulliUCTPolicy, UCTParams(trials=256, maxdepth=4, C=10)),
}

games = {
    'total': TotalRegretGame,
    'simple': SimpleRegretGame
}

filename = sys.argv[1]
game_type = sys.argv[2]

N = 100
T = 200

Ks = [2]
deltas = [0.15, 0.25, 0.35]

fig = plt.figure(figsize=(20, 10))

i = 1
for delta in deltas:

    for K in Ks:

        probs = [0.5] * (K - 1) + [0.5 + delta]
        bandit = BernoulliBandit(probs)

        print('\n>> Bernoulli(K={}, delta={}) ...\n'.format(K, delta))

        results = {}

        for name, policy in policies.items():
            print('{} is playing ... '.format(name), end='')

            start = time.time()

            if isinstance(policy, tuple):
                learner = policy[0](bandit.size, T, policy[1])
            else:
                learner = policy(bandit.size, T)

            game = games[game_type](bandit, learner)
            results[name] = game.run(N, T)

            end = time.time()
            uptime = end - start
            print('done in {:.4f} sec.'.format(uptime))

        plot_results(game_type, results, deltas, Ks, i, T)

        i += 1


plt.savefig('{}.pdf'.format(filename), format='pdf')
