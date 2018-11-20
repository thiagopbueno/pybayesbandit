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

import numpy as np


class BetaBernoulliMDP(BeliefMDP):

    def __init__(self, actions):
        self.actions = actions

    @property
    def start(self):
        '''Uniform prior'''
        return tuple([(1, 1)] * self.actions)

    def sample(self, belief, action):
        '''
        Sample from belief-state transition.
        '''
        alpha, beta = belief[action]
        theta = alpha / (alpha + beta)
        r = int(np.random.sample() >= theta)
        next_belief = tuple((params[0] + r, params[1] + 1 - r) if i == action else params \
            for i, params in enumerate(belief))
        return next_belief

    # def sample(self, belief, action):
    #     '''
    #     Sample bandit from posterior and then sample reward and
    #     return corresponding next belief-state
    #     '''
    #     alpha, beta = belief[action]
    #     theta = np.random.beta(alpha, beta)
    #     r = int(np.random.sample() >= theta)
    #     next_belief = tuple((params[0] + r, params[1] + 1 - r) if i == action else params \
    #             for i, params in enumerate(belief))
    #     return next_belief

    def transition(self, belief, action):
        alpha, beta = belief[action]

        probs_and_next_beliefs = []
        for r in [0, 1]:
            next_belief = tuple((params[0] + r, params[1] + 1 - r) if i == action else params \
                for i, params in enumerate(belief))
            prob = (r * alpha + (1 - r) * beta) / (alpha + beta)
            probs_and_next_beliefs.append((prob, next_belief))

        return probs_and_next_beliefs

    # def reward(self, belief, action, next_belief):
    #     '''Expected Value of Information (EVOI)'''
    #     eu_belief = max(alpha / (alpha + beta) for alpha, beta in belief)
    #     eu_next_belief = max(alpha / (alpha + beta) for alpha, beta in next_belief)
    #     evof = eu_next_belief - eu_belief
    #     return evof

    # def reward(self, belief, action, next_belief):
    #     '''Maximum Expeted Utility (MEU[b_t])'''
    #     return max(alpha / (alpha + beta) for alpha, beta in belief)

    def reward(self, belief, action, next_belief):
        '''Expected Utility (EU[b_t, a_t])'''
        alpha, beta = belief[action]
        return alpha / (alpha + beta)

    # def reward(self, belief, action, next_belief):
    #     '''Beta/Bernoulli payoff'''
    #     alpha, beta = belief[action]
    #     next_alpha, next_beta = next_belief[action]
    #     if next_alpha - alpha == 1 and next_beta - beta == 0:
    #         return 1
    #     if next_alpha - alpha == 0 and next_beta - beta == 1:
    #         return 0
