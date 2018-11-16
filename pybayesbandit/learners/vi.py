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

import sys


class ValueIteration():

    def __init__(self, mdp):
        self._mdp = mdp
        self._V = {}
        
    def __call__(self, T):
        self.V(self._mdp.start, T)
        return self._V

    def V(self, belief, T):
        if T == 0:
            return (None, 0.0)

        if (T, belief) in self._V:
            return self._V[(T, belief)]

        action, value = None, -sys.maxsize
        for a in range(self._mdp.actions):
            Q = self.Q(a, belief, T)
            if Q > value:
                action = a
                value = Q

        self._V[(T, belief)] = (action, value)
        return (action, value)

    def Q(self, action, belief, T):
        q_value = 0.0
        for prob, next_belief in self._mdp.transition(belief, action):
            q_value += prob * (self._mdp.reward(belief, action, next_belief) + self.V(next_belief, T - 1)[1])
        return q_value


class BetaBernoulliVIPolicy(Learner):

    def __init__(self, actions, T, params=None):
        self.actions = actions
        self.T = T
        self._solve()
        self.reset()

    def _solve(self):
        self._mdp = BetaBernoulliMDP(self.actions)
        self._vi = ValueIteration(self._mdp)
        self._V = self._vi(self.T)

    def __call__(self):
        action, _ = self._V[(self._T, self._belief)]
        return action

    def update(self, action, reward):
        alpha, beta = self._belief[action]
        self._belief = tuple((params[0] + reward, params[1] + 1 - reward) if i == action else params \
                for i, params in enumerate(self._belief))
        self._T -= 1

    def reset(self):
        self._T = self.T
        self._belief = self._mdp.start
