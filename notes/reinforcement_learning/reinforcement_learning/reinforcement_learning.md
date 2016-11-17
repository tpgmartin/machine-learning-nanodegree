# Reinforcement Learning

(in Markov decision processes)

Previously

Model (T,R) -- Planner --> Policy (Pi) "Planning approach"

Now

Transitions (s,a,r,s',s*) -- Learning --> Policy "Reinforcement learning"

## Rat Dinosaurs

Reward maximisation through action selection

Transitions -- Modeler --> Model

Model -- Simulator --> Transitions

## API

Model-based reinforcement learning
vs
Reinforcement-based planner

## Three Approaches to RL

Policy search: s -- Pi --> a "direct use, indirect learning"

Value-function based: s -- U --> v (transform utility to policy with argmax)

Model based: s,a -- T, R --> s',r' (solve Bellman equations to obtain utility)
"direct learning, indirect use"

## Learning Incrementally

V converges to E[X]

## Q Learning Convergence

[Convergence of Q-Learning](http://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf)

## Choosing Actions

Q-Learning is a family of algorithms

"Simulated annealing"-like approach take a random action occasionally

## Epsilon-Greedy Exploration

Exploration-Exploitation dilemma i.e. interatctiong between machine learning 
and planning

[Reinforcement Learning: A Story](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume4/kaelbling96a.pdf) 
