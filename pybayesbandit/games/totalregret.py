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

from pybayesbandit.games import Game

import numpy as np


class TotalRegretGame(Game):

    def __init__(self, bandit, learner):
        super().__init__(bandit, learner)

    def episode(self, T):
        self.learner.reset()

        actions = np.zeros(T, dtype=np.float32)
        rewards = np.zeros(T, dtype=np.float32)
        regrets = np.zeros(T, dtype=np.float32)

        for t in range(T):
            a = self.learner()
            r = self.bandit(a)
            self.learner.update(a, r)

            actions[t] = a
            rewards[t] = r
            regrets[t] = self.bandit.regret(a)

        return actions, rewards, regrets

    def run(self, N, T):
        total_rewards = np.zeros([N, T], dtype=np.float32)
        total_regrets = np.zeros([N, T], dtype=np.float32)

        for n in range(N):
            actions, rewards, regrets = self.episode(T)
            total_rewards[n] = np.cumsum(rewards, axis=0)
            total_regrets[n] = np.cumsum(regrets, axis=0)

        avg_total_rewards = np.mean(total_rewards, axis=0)
        std_total_rewards = np.std(total_rewards, axis=0)

        avg_total_regrets = np.mean(total_regrets, axis=0)
        std_total_regrets = np.std(total_regrets, axis=0)

        return (avg_total_rewards, std_total_rewards), (avg_total_regrets, std_total_regrets)
