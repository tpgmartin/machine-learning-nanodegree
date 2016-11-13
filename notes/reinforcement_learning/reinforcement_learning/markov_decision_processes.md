# Markov Decision Processes

## Decision Making and Reinforcement Learning

Supervised learning: y = f(x) function approximation
Unsupervised learning: f(x) clustering description
Reinforcement learning: y = f(x) different character to function approximation

## Markov Decision Processes

States:  S
Model:   T(s,a,s') ~ Pr(s'|s,a)
Actions: A(s), A
Reward:  R(s), R(s,a), R(s,a,s')
--------------------------------
Policy:  Pi(s) -> a
         Pi*(s) maximal reward expected to receive at any given state

* Actions what you can do in a particular state
* Model transition state from state s to state s', given actions a
* Reward can be defined multiple ways, defines usefulness of entering a given 
  state
* Policy, takes in state, returns an action. A solution to Markov problem
  - Map each state to action
  - Will always know state and action
* Markov
  - only present state matters
  - stationary, rules defined above do not change with time
  - defined by state, model, action, and reward above

## More About Rewards

* Delayed reward - unclear what success is until very end of sequence of 
actions
* (Temporal) credit assignment

R(s) = -0.04 with end states +1, -1 encourages finishing, but not as quickly as 
possible, in positive end state as reward for action is both very small and 
negative

Conversely: R(s) = +2 encourages prolonging game indefinitely

Conversely: R(s) = -2 encourages finishing game as quickly as possible in 
either end state

## Sequence of Rewards

Assumptions: stationarity
* Infinite horizons: policy can change even if in same state i.e. Pi(s,t) -> a
* Utility of sequences i.e. if U(s_0, s_1, s_2) > U(s'_0, s'_1, s'_2) then 
  U(s_1, s_2) > U(s'_1, s'_2) "stationary preferences". This has to be true to
  guarantee stationarity assumption

U(s_0, s_1, s_2) = Sum of rewards over all states

Utility of infinite sequence of states implies no single reward function is 
better than any other

## Policies

Bellman Equation: fundamental recursive equation defining utility for state
- Solve with value iteration or policy iteration

(value functions or utilities - long term)

A proof of the convergence of value iteration shown on [slide 25 of slides](https://s3.amazonaws.com/ml-class/notes/MDPIntro.pdf)


