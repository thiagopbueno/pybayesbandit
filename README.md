# pybayesbandit

Bayesian bandits in Python3.

## Quickstart

```text
$ pip install pybayesbandit
```

## Usage

```text
$ pybayesbandit --help

usage: pybayesbandit [-h] [-p PARAMS [PARAMS ...]] [-d MAXDEPTH] [-t TRIALS]
                     [-e EPISODES] [-hr HORIZON] [-v]
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
  -d MAXDEPTH, --maxdepth MAXDEPTH
                        maximum number of timesteps in the tree lookahead
                        (default=10)
  -t TRIALS, --trials TRIALS
                        number of trials in Monte-Carlo sampling (default=30)
  -e EPISODES, --episodes EPISODES
                        number of simulation episodes (default=200)
  -hr HORIZON, --horizon HORIZON
                        number of timesteps in each episode (default=100)
  -v, --verbose         verbose mode
```

## Examples

```text
$ pybayesbandit ucb bernoulli -p 0.5 0.8 0.3 -e 100 -hr 50 -v

Running pybayesbandit ...
>> learner  = ucb
>> bandit   = bernoulli([0.5, 0.8, 0.3])
>> episodes = 100
>> horizon  = 50
Done in 0.208 sec.

Results:
>> Reward =  32.4200 ± 3.1912
>> Regret =   7.5990 ± 1.4146
```

```text
$ pybayesbandit thompson bernoulli -p 0.5 0.8 0.3 -e 100 -hr 50 -v

Running pybayesbandit ...
>> learner  = thompson
>> bandit   = bernoulli([0.5, 0.8, 0.3])
>> episodes = 100
>> horizon  = 50
Done in 0.213 sec.

Results:
>> Reward =  35.4500 ± 3.6943
>> Regret =   4.3990 ± 2.2248
```

```text
$ pybayesbandit uct bernoulli -p 0.5 0.8 0.3 -e 100 -hr 50 --trials 15 --maxdepth 5 -v

Running pybayesbandit ...
>> learner  = uct(trials=15, maxdepth=5)
>> bandit   = bernoulli([0.5, 0.8, 0.3])
>> episodes = 100
>> horizon  = 50
Done in 25.503 sec.

Results:
>> Reward =  36.7400 ± 5.3191
>> Regret =   3.2210 ± 5.1306
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
