
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


class ThompsonSamplingPolicy(Learner):

    def __init__(self, actions, T, params=None):
        self.actions = actions
        self.reset()

    def __call__(self):
        samples = [np.random.beta(alpha, beta) for alpha, beta in self.betas]
        return np.argmax(samples)

    def update(self, action, reward):
        alpha, beta = self.betas[action]
        if reward == 1:
            alpha += 1
        else:
            beta += 1
        self.betas[action] = (alpha, beta)

    def reset(self):
        self.betas = [(1.0, 1.0)] * self.actions
