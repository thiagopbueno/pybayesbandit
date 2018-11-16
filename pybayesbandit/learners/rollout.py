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
from pybayesbandit.mdp.beta_bernoulli import BetaBernoulliMDP

import numpy as np
import sys


class RolloutPolicy(Learner):

    def __init__(self, actions, T, params=None):
        self.actions = actions
        self.T = T
        self.trials = params.trials
        self.mdp = BetaBernoulliMDP(self.actions)
        self.reset()

    def __call__(self):
        best_action = None
        best_q_value = -sys.maxsize

        for action in range(self.actions):
            q = self._q_value(self._belief, action, self._T)

            if q > best_q_value:
                best_q_value = q
                best_action = action

        return best_action

    def update(self, action, reward):
        alpha, beta = self._belief[action]
        self._belief = tuple((params[0] + reward, params[1] + 1 - reward) if i == action else params \
                for i, params in enumerate(self._belief))
        self._T -= 1

    def reset(self):
        self._T = self.T
        self._belief = self.mdp.start

    def _rollout(self, state, action, depth):
        total = 0.0

        next_state = self.mdp.sample(state, action)
        total += self.mdp.reward(state, action, next_state)

        for step in range(depth-1):
            state = next_state
            action = np.random.randint(0, self.mdp.actions)
            next_state = self.mdp.sample(state, action)
            total += self.mdp.reward(state, action, next_state)

        return total

    def _q_value(self, state, action, depth):
        q = 0.0
        for _ in range(self.trials):
            q += self._rollout(state, action, depth)
        q /= self.trials
        return q

