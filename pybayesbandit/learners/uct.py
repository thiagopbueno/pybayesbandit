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


class Node():

    def __init__(self, state, action=None, visits=0, value=0.0, succ=None):
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
        return hash(self) == hash(other)


class UCT():

    def __init__(self, mdp, start):
        self._mdp = mdp
        self._start = start
        self.nodes = {}

    def __call__(self, T, trials=100):
        self._trials = trials
        start = tuple(self._start)
        n0 = Node(start)
        self.nodes[n0] = n0
        for _ in range(trials):
            self.rollout(n0, T)
        values = [a.value for a in n0.succ]
        return n0.succ[np.argmax(values)].action

    def rollout(self, node, depth):
        r = 0.0

        if node.is_decision():
            if node.succ is None:
                node.succ = []
                total_visits = 0
                total_value = 0
                for a in range(self._mdp.actions):
                    visits, value = self._heuristics(node, a, depth)
                    total_visits += visits
                    total_value += value
                    n = Node(node.state, a, visits, value)
                    self.nodes[n] = n
                    node.succ.append(n)
                node.visits = total_visits
                node.value = total_value / self._mdp.actions
            next_node = self.ucb(node)
        else:
            state = list(node.state)
            action = node.action
            next_state = self._mdp.sample(state, action)
            r = self._mdp.reward(state, action, next_state)
            depth -= 1
            if depth == 0:
                return r
            next_node = Node(tuple(next_state))
            if next_node in self.nodes:
                next_node = self.nodes[next_node]
            else:
                self.nodes[next_node] = next_node

        r += self.rollout(next_node, depth)
        self.update(node, r)
        return r

    def ucb(self, node):
        for a in node.succ:
            if a.visits == 0:
                return a
        bounds = [a.value + np.sqrt(2 * np.log(node.visits) / a.visits) for a in node.succ]
        index = np.argmax(bounds)
        return node.succ[index]

    def update(self, node, r):
        node.visits += 1
        node.value += 1 / node.visits * (r - node.value)

    def _heuristics(self, node, a, T):
        alpha, beta = node.state[a]
        value = T * alpha / (alpha + beta)
        visits = 0.1 * self._trials
        return visits, value


class BetaBernoulliUCTPolicy(Learner):

    def __init__(self, actions, T):
        self.actions = actions
        self.T = T
        self._mdp = BetaBernoulliMDP(self.actions)
        self.reset()

    def __call__(self):
        uct = UCT(self._mdp, self._belief)
        return uct(self._T, trials=30)

    def update(self, action, reward):
        alpha, beta = self._belief[action]
        self._belief[action] = (alpha + reward, beta + 1 - reward)
        self._T -= 1

    def reset(self):
        self._T = self.T
        self._belief = self._mdp.start
