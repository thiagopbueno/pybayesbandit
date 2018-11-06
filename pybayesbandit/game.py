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


import numpy as np


class Game():

    def __init__(self, bandit, learner):
        self.bandit = bandit
        self.learner = learner

    def episode(self, T):
        self.learner.reset()
        rewards = []
        actions = []
        for t in range(T):
            a = self.learner()
            r = self.bandit(a)
            self.learner.update(a, r)
            actions.append(a)
            rewards.append(r)
        total = np.sum(rewards)
        regret = self.bandit.regret(actions)
        return total, regret

    def run(self, N, T):
        regrets = []
        totals = []
        for n in range(N):
            total, regret = self.episode(T)
            totals.append(total)
            regrets.append(regret)
        avg_total_reward = np.mean(totals)
        stddev_total_reward = np.std(totals)
        avg_regret = np.mean(regrets)
        stddev_regret = np.std(regrets)
        return avg_total_reward, stddev_total_reward, avg_regret, stddev_regret
