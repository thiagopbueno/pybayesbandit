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
                     [-C C] [-e EPISODES] [-hr HORIZON] [--plot] [-v]
                     {random,ucb,thompson,vi,uct,rollout,aotree} {bernoulli}
                     {total,simple}

Bayesian bandits in Python3.

positional arguments:
  {random,ucb,thompson,vi,uct,rollout,aotree}
                        learner type
  {bernoulli}           bandit type
  {total,simple}        game setting

optional arguments:
  -h, --help            show this help message and exit
  -p PARAMS [PARAMS ...], --params PARAMS [PARAMS ...]
                        bandit parameters
  -d MAXDEPTH, --maxdepth MAXDEPTH
                        maximum number of timesteps in the tree lookahead
                        (default=10)
  -t TRIALS, --trials TRIALS
                        number of trials in Monte-Carlo sampling (default=30)
  -C C                  UCT exploration constant (default=2.0)
  -e EPISODES, --episodes EPISODES
                        number of simulation episodes (default=200)
  -hr HORIZON, --horizon HORIZON
                        number of timesteps in each episode (default=100)
  --plot                plot cumulative regret
  -v, --verbose         verbose mode
```

## Examples

```text
$ pybayesbandit ucb bernoulli total -p 0.5 0.8 0.3 -e 100 -hr 50 -v

Running pybayesbandit ...
>> learner  = ucb
>> bandit   = bernoulli([0.5, 0.8, 0.3])
>> episodes = 100
>> horizon  = 50
Done in 0.257 sec.

Results:
>> Reward =  32.3900 ± 3.4173
>> Regret =   7.6530 ± 1.7081
```

```text
$ pybayesbandit thompson bernoulli total -p 0.5 0.8 0.3 -e 100 -hr 50 -v

Running pybayesbandit ...
>> learner  = thompson
>> bandit   = bernoulli([0.5, 0.8, 0.3])
>> episodes = 100
>> horizon  = 50
Done in 0.297 sec.

Results:
>> Reward =  35.2200 ± 3.8822
>> Regret =   4.4560 ± 2.6086
```

```text
$ pybayesbandit uct bernoulli total -p 0.5 0.8 0.3 -e 100 -hr 50 --trials 15 --maxdepth 5 -v

Running pybayesbandit ...
>> learner  = uct(trials=15, maxdepth=5, C=2.0)
>> bandit   = bernoulli([0.5, 0.8, 0.3])
>> episodes = 100
>> horizon  = 50
Done in 7.066 sec.

Results:
>> Reward =  36.2800 ± 5.4828
>> Regret =   3.4360 ± 4.5856
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
