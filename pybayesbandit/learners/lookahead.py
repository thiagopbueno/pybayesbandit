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
from pybayesbandit.search.aotree import AOTreeSearch

import sys


class OptimisticLimitedDepthAOTree(AOTreeSearch):

    def __init__(self, mdp):
        super().__init__(mdp)

    def heuristic(self, state):
        best_action, best_mean = None, -sys.maxsize
        for i, (alpha, beta) in enumerate(state):
            mean = alpha / (alpha + beta)
            if mean > best_mean:
                best_action = i
                best_mean = mean
        return (best_action, (self.horizon - self.max_depth) * best_mean)


class LookaheadTreeSearchPolicy(Learner):

    def __init__(self, actions, T, params=None):
        self.actions = actions
        self.T = T
        self.max_depth = params.maxdepth
        self.mdp = BetaBernoulliMDP(self.actions)
        self.aotree = OptimisticLimitedDepthAOTree(self.mdp)
        self.reset()

    def __call__(self):
        depth = min(self.horizon, self.max_depth)
        action = self.aotree(self.belief, depth, self.horizon)
        return action

    def update(self, action, reward):
        alpha, beta = self.belief[action]
        self.belief = tuple((params[0] + reward, params[1] + 1 - reward) if i == action else params \
                for i, params in enumerate(self.belief))
        self.horizon -= 1

    def reset(self):
        self.horizon = self.T
        self.belief = self.mdp.start
