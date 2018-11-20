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


class Node():

    def __init__(self, state, action=None, visits=0.0, value=0.0, succ=None):
        self.state = state
        self.action = action
        self.visits = visits
        self.value = value
        self.succ = succ if succ is not None else []

    def is_decision(self):
        return self.action is None

    def is_chance(self):
        return self.action is not None

    def is_leaf(self):
        return self.succ == []

    def __hash__(self):
        return hash((self.state, self.action))

    def __eq__(self, other):
        return self.state == other.state and self.action == other.action

    def __repr__(self):
        return 'Node(state={}, action={}, visits={}, value={}, succ={})'.format(self.state, self.action, self.visits, self.value, self.succ)


class MCTS(metaclass=abc.ABCMeta):

    def __init__(self, mdp):
        self.mdp = mdp
        self.nodes = {}

    @abc.abstractmethod
    def tree_policy(self, node):
        raise NotImplementedError

    @abc.abstractmethod
    def default_policy(self, state):
        raise NotImplementedError

    @abc.abstractmethod
    def init_q_value(self, node, d):
        raise NotImplementedError

    @abc.abstractmethod
    def heuristic(self, state):
        raise NotImplementedError

    def __call__(self, start, max_depth, horizon, trials, C=2):
        self.max_depth = max_depth
        self.horizon = horizon

        n0 = Node(start)
        self.nodes[n0] = n0

        for i in range(trials):
            self._trial(n0, max_depth, C)

        best_q_value = -sys.maxsize
        best_action_node = None
        for a in n0.succ:
            if a.value > best_q_value:
                best_q_value = a.value
                best_action_node = a

        return best_action_node.action

    def _trial(self, node, depth, C):
        r = 0.0
        if depth == 0:
            return self.heuristic(node.state)

        if node.is_decision():

            if node.is_leaf(): # expand decision node
                for action in range(self.mdp.actions):
                    succ_node = Node(node.state, action)
                    node.succ.append(succ_node)
                next_node = node.succ[0]
            else: # traverse tree
                next_node = self.tree_policy(node, C)

        else: # node.is_chance() == True

            if node.visits == 0: # initialize leaf node and backup
                # r = self._rollout(node.state, node.action, depth)
                r = self.init_q_value(node, depth)
                node.visits = 1
                node.value = r
                return r
            else: # traverse tree

                state = node.state
                action = node.action
                next_state = self.mdp.sample(state, action)
                r = self.mdp.reward(state, action, next_state)

                next_node = Node(next_state)
                if next_node in self.nodes: # traverse tree
                    next_node = self.nodes[next_node]
                else: # expand chance node
                    self.nodes[next_node] = next_node

                depth -= 1

        r += self._trial(next_node, depth, C)
        self._backup(node, r)

        return r

    # def _rollout(self, state, action, depth):
    #     total = 0.0

    #     next_state = self.mdp.sample(state, action)
    #     total += self.mdp.reward(state, action, next_state)

    #     for step in range(depth-1):
    #         state = next_state
    #         action = self.default_policy(state)
    #         next_state = self.mdp.sample(state, action)
    #         total += self.mdp.reward(state, action, next_state)

    #     return total

    def _backup(self, node, r):
        node.visits += 1
        node.value += 1 / node.visits * (r - node.value)
