# pybayesbandit

Bayesian bandits in Python3.

## Quickstart

```text
$ pip install pybayesbandit
```

## Usage

```text
$ pybayesbandit --help

usage: pybayesbandit [-h] [-p PARAMS [PARAMS ...]] [-e EPISODES] [-hr HORIZON]
                     [-v]
                     {random,ucb,thompson,vi,uct} {bernoulli}

Bayesian bandits in Python3.

positional arguments:
  {random,ucb,thompson,vi,uct}
                        learner type
  {bernoulli}           bandit type

optional arguments:
  -h, --help            show this help message and exit
  -p PARAMS [PARAMS ...], --params PARAMS [PARAMS ...]
                        bandit parameters
  -e EPISODES, --episodes EPISODES
                        number of simulation episodes (default=200)
  -hr HORIZON, --horizon HORIZON
                        number of timesteps in each episode (default=100)
  -v, --verbose         verbose mode
```

## Examples

```text
$ pybayesbandit thompson bernoulli -p 0.5 0.8 0.3 -e 100 -hr 20 -v

Running pybayesbandit ...
>> learner  = thompson
>> bandit   = bernoulli([0.5, 0.8, 0.3])
>> episodes = 100
>> horizon  = 20
Done in 0.094 sec.

Results:
>> Reward =  13.2800 ± 2.3541
>> Regret =   3.2760 ± 1.7144
```

## License

Copyright (c) 2018 Thiago Pereira Bueno All Rights Reserved.

pybayesbandit is free software: you can redistribute it and/or modify it
under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

pybayesbandit is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with pybayesbandit. If not, see http://www.gnu.org/licenses/.
