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


from pybayesbandit.mdp import BeliefMDP


class BetaBernoulliMDP(BeliefMDP):

    def __init__(self, actions):
        self.actions = actions

    @property
    def start(self):
        return [(1, 1)] * self.actions

    def sample(self, belief, action):
        probs, next_beliefs = zip(*self.transition(belief, action))
        i = np.squeeze(np.where(np.random.multinomial(1, probs) == 1))
        return next_beliefs[i]

    def transition(self, belief, action):
        alpha, beta = belief[action]

        probs_and_next_beliefs = []
        for r in [0, 1]:
            next_belief = belief.copy()
            next_belief[action] = (alpha + r, beta + 1 - r)
            prob = (r * alpha + (1 - r) * beta) / (alpha + beta)
            probs_and_next_beliefs.append((prob, next_belief))

        return probs_and_next_beliefs

    def reward(self, belief, action, next_belief):
        alpha, beta = belief[action]
        next_alpha, next_beta = next_belief[action]
        if next_alpha - alpha == 1 and next_beta - beta == 0:
            return 1
        if next_alpha - alpha == 0 and next_beta - beta == 1:
            return 0
