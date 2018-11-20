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

import abc
import sys


class AOTreeSearch(metaclass=abc.ABCMeta):

    def __init__(self, mdp):
        self.mdp = mdp

    @abc.abstractmethod
    def heuristic(self, state):
        raise NotImplementedError

    def __call__(self, start, max_depth, horizon):
        self.V = {}
        self.max_depth = max_depth
        self.horizon = horizon
        action, _ = self.or_node(start, max_depth)
        return action

    def or_node(self, state, depth):
        if depth == 0:
            return self.heuristic(state)

        if (depth, state) in self.V:
            return self.V[(depth, state)]

        best_action, best_value = None, -sys.maxsize
        for action in range(self.mdp.actions):
            Q = self.and_node(state, action, depth)
            if Q > best_value:
                best_action = action
                best_value = Q

        self.V[(depth, state)] = (best_action, best_value)

        return (best_action, best_value)

    def and_node(self, state, action, depth):
        q_value = 0.0
        for prob, next_state in self.mdp.transition(state, action):
            q_value += prob * (self.mdp.reward(state, action, next_state) + self.or_node(next_state, depth - 1)[1])
        return q_value
