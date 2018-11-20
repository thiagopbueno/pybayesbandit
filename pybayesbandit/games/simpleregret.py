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


class SimpleRegretGame(Game):

    def __init__(self, bandit, learner):
        super().__init__(bandit, learner)

    def episode(self, T):
        self.learner.reset()

        for t in range(T-1):
            a = self.learner()
            r = self.bandit(a)
            self.learner.update(a, r)

        a = self.learner()
        regret = self.bandit.regret(a)

        return regret

    def run(self, N, T):
        simple_regrets = np.zeros([N], dtype=np.float32)

        for n in range(N):
            regret = self.episode(T)
            simple_regrets[n] = regret

        avg_simple_regrets = np.mean(simple_regrets, axis=0)
        std_simple_regrets = np.std(simple_regrets, axis=0)

        return (avg_simple_regrets, std_simple_regrets)
