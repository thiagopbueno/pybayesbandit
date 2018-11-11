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


from pybayesbandit.learners import Learner

import numpy as np


class UCBPolicy(Learner):

    def __init__(self, actions, T, params=None):
        self.actions = actions
        self.reset()

    def __call__(self):
        if self.n < self.actions:
            return self.n
        else:
            return np.argmax(self.avg + np.sqrt(2 * np.log(self.n) / self.counts))

    def update(self, action, reward):
        self.n += 1
        self.counts[action] += 1
        self.avg[action] = self.avg[action] + (1 / self.counts[action]) * (reward - self.avg[action])

    def reset(self):
        self.avg = np.zeros(self.actions)
        self.counts = np.zeros(self.actions)
        self.n = 0
