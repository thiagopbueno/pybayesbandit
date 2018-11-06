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


class BeliefMDP(metaclass=abc.ABCMeta):

    @abc.abstractproperty
    def start(self):
        raise NotImplemented

    @abc.abstractmethod
    def sample(self, belief, action):
        raise NotImplemented

    @abc.abstractmethod
    def transition(self, belief, action):
        raise NotImplemented

    @abc.abstractmethod
    def reward(self, belief, action, next_belief):
        raise NotImplemented
