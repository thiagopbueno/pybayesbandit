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


class Node():

    def __init__(self, state, action=None, visits=None, value=None, succ=None):
        self.state = state
        self.action = action
        self.visits = visits
        self.value = value
        self.succ = succ

    def is_decision(self):
        return self.action is None

    def is_chance(self):
        return self.action is not None

    def __hash__(self):
        return hash((self.state, self.action))

    def __eq__(self, other):
        return self.state == other.state and self.action == other.action

    def __repr__(self):
        return 'Node(state={}, action={}, visits={}, value={}, succ={})'.format(self.state, self.action, self.visits, self.value, self.succ)


class UCT():

    def __init__(self, mdp, start, trials=100):
        self._mdp = mdp
        self._start = start
        self.trials = trials
        self.nodes = {}

    def __call__(self, maxdepth):
        start = tuple(self._start)
        n0 = Node(start)
        n0.visits = 0
        n0.value = self._init_V_fn(n0, maxdepth)
        n0.succ = []
        self.nodes[n0] = n0
        for _ in range(self.trials):
            self.rollout(n0, maxdepth)
        values = [a.value for a in n0.succ]
        return n0.succ[np.argmax(values)].action

    def rollout(self, node, depth):
        res = 0.0

        if node.is_decision():
            if self._decision_node_has_unvisited_successor(node):
                next_node = self._get_decision_node_unvisited_successor(node, depth)
                node.succ.append(next_node)
                self.nodes[next_node] = next_node
            else:
                next_node = self.ucb(node)
        else:
            state = list(node.state)
            action = node.action
            next_state = self._mdp.sample(state, action)
            res = self._mdp.reward(state, action, next_state)

            depth -= 1
            if depth == 0:
                return res

            next_node = Node(tuple(next_state))
            if next_node in self.nodes:
                next_node = self.nodes[next_node]
            else:
                next_node.visits = 0
                next_node.value = self._init_V_fn(next_node, depth)
                next_node.succ = []
                node.succ.append(next_node)
                self.nodes[next_node] = next_node

        res += self.rollout(next_node, depth)
        self.update(node, res)

        return res

    def ucb(self, node):
        best_action = None
        best_Q = -sys.maxsize
        for a in node.succ:
            if a.visits == 0:
                return a
            q = a.value + np.sqrt(2 * np.log(node.visits) / a.visits)
            if q > best_Q:
                best_Q = q
                best_action = a
        return best_action

    def update(self, node, r):
        node.visits += 1
        node.value += 1 / node.visits * (r - node.value)

    def _decision_node_has_unvisited_successor(self, node):
        return (node.succ is None or len(node.succ) != self._mdp.actions)

    def _get_decision_node_unvisited_successor(self, node, depth):
        state = node.state
        action = len(node.succ) if node.succ is not None else 0
        visits = 1
        value = self._init_Q_fn(node, action, depth)
        succ = []
        return Node(state, action, visits, value, succ)

    def _init_Q_fn(self, node, a, d):
        alpha, beta = node.state[a]
        h = d * alpha / (alpha + beta)
        return h

    def _init_V_fn(self, node, d):
        return 0.0


class BetaBernoulliUCTPolicy(Learner):

    def __init__(self, actions, T, params):
        self.actions = actions
        self.T = T
        self.trials = params.trials
        self.maxdepth = params.maxdepth
        self._mdp = BetaBernoulliMDP(self.actions)
        self.reset()

    def __call__(self):
        uct = UCT(self._mdp, self._belief, self.trials)
        depth = min(self._step, self.maxdepth)
        return uct(depth)

    def update(self, action, reward):
        alpha, beta = self._belief[action]
        self._belief[action] = (alpha + reward, beta + 1 - reward)
        self._step -= 1

    def reset(self):
        self._step = self.T
        self._belief = self._mdp.start
