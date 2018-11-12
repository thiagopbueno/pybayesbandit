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


from pybayesbandit.bandits import Bandit

import numpy as np


class BernoulliBandit(Bandit):

    def __init__(self, probs):
        self._probs = probs
        self._optimal = np.max(probs)

    @property
    def size(self):
        return len(self._probs)

    def __call__(self, action):
        assert 0 <= action < self.size
        return np.random.binomial(1, self._probs[action])

    def regret(self, action):
        return self._optimal - self._probs[action]

    def total_regret(self, actions):
        T = len(actions)
        return T * self._optimal - np.sum(self._probs[a] for a in actions)
