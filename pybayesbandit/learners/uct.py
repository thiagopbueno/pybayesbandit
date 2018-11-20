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
from pybayesbandit.search.mcts import MCTS

import numpy as np
import sys


class UCT(MCTS):

    def __init__(self, mdp):
        super().__init__(mdp)
        self.h = {}

    def tree_policy(self, node, C=2):
        best_action = None
        best_q_value = -sys.maxsize

        for a in node.succ:
            if a.visits == 0:
                return a

            q = a.value + C * np.sqrt(2 * np.log(node.visits) / a.visits)

            if q > best_q_value:
                best_q_value = q
                best_action = a

        return best_action

    def default_policy(self, state):
        return np.random.randint(0, self.mdp.actions)

    def init_q_value(self, node, d):
        alpha, beta = node.state[node.action]
        h = (self.horizon - d) * alpha / (alpha + beta)
        return h

    def heuristic(self, state):
        if state in self.h:
            return self.h[state]
        best_mean = max(alpha / (alpha + beta) for (alpha, beta) in state)
        h = (self.horizon - self.max_depth) * best_mean
        self.h[state] = h
        return h


class BetaBernoulliUCTPolicy(Learner):

    def __init__(self, actions, T, params):
        self.actions = actions
        self.T = T
        self.trials = params.trials
        self.max_depth = params.maxdepth
        self.C = params.C
        self.mdp = BetaBernoulliMDP(self.actions)
        self.reset()

    def __call__(self):
        depth = min(self._step, self.max_depth)
        self.uct = UCT(self.mdp)
        action = self.uct(self._belief, depth, self.T, self.trials, self.C)
        return action

    def update(self, action, reward):
        alpha, beta = self._belief[action]
        self._belief = tuple((params[0] + reward, params[1] + 1 - reward) if i == action else params \
                for i, params in enumerate(self._belief))
        self._step -= 1

    def reset(self):
        self._step = self.T
        self._belief = self.mdp.start
