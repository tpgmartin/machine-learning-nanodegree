# Game Theory

## What is it?

* Mathematics of conflict
* Game theory allows us to move from single agents to multiple agents
* Economics and politics
* Increasingly a part of AI/ML

## A Simple Game

2-player zero-sum finite deterministic game of perfect information:

Binary tree, agents choose actions (L,R) to move between states, leaves equal 
a's + reward, b's - reward

(Pure) Strategies, like policies in RL:

|   |   |   |       B       |
|   |   |   | 2 | L | M | R |
|   |   |   | 3 | R | R | R |
|   | 1 | 4 |   |   |   |   |
|   | L | L |   | 7 | 3 | -1 |
| A | L | R |   | 7 | 3 | 4 |
|   | R | L |   | 2 | 2 | 2 |
|   | R | R |   | 2 | 2 | 2 |

## Minmax

Strategy (Minmax):
* 'A' considers worst case counter action - maximise score
* 'B' considers worst case counter action - minimise score

Max value for A, B is 3 (from matrix above)

## Fundamental Result

In a 2-player, zero-sum deterministic game of perfect information: Minimum is 
equivalent to maximum, and there always exists an optimal pure strategy for 
each player

* optimal = all players trying to maximise score

Can construct tree consistent with matrix

## Game tree

Von Neumann theorem: other theorem from deterministic game still holds

Case of 2-playerm zero-sum non-deterministic game of perfet information 

Matrix contains all information of given tree, can reconstruct a number of 
trees from one given matrix.

Example matrix

|   |   |  B |    |
|   |   |  L |  R |
| A | L | -8 | -8 |
|   | R | -2 |  3 |

Solution will be -2 as A will want to maximise, B will want to minimise

## Minipoker

2-player, zero-sum, non-deterministic game of hidden information

- A is dealt a card, red or black 50%
- A may resign if red: -20 cents for A else A holds
  - B resigns: +10 cents
  - B sees:
    - If red: -40 cents
    - If black: +30 cents

## Minipoker tree

Hidden as B does not know apriori what state it is in

|   |          |      B     |          |
|   |          |  Resigner  |    Seer  |
| A | Resigner |     -5     |     5    |
|   | Holder   |     10     |    -5    |

In this case minimum not equal to maximum, Von Neumann theorem does not hold

## Mixed Strategy

Difference from pure strategy, probabilistically choose between strategies
Mixed strategy = Distribution over strategies
Probability, P = probability of being a holder

Determine payout for given agent in terms of P, wrt different strategies for 
other agent

Expected value is found from intersection of lines

## Snitch

2-player, non-zero sum, non-deterministic game of hidden information 

|   |          |      B       |            |
|   |          |     Coop     |    Defect  |
| A | Coop     |     -1,-1    |    -9,0    |
|   | Defect   |     0,-9     |    -6,-6   |

Best choice for individual agent is to defect regardless of other agent's 
choice

## A Beautiful Equilibrium

n players with strategies S_1, S_2, ..., S_n

S^*_1 in S_1, ..., S^*_n in S_n

are a Nash Equilibrium iff 

For all i, S^*_i = argmax_s_i utility_i(S^*_1,...,S^*_n)

i.e. no reason for anyone to change

Strictly dominated equilibrium

Consequences:

* In the n-player, pure strategy game, if equilibrium of strictly dominated 
  strategies eliminates all but one combintation, that combination is the 
  unique Nash equilibrium (NE)
* Any NE will survive elimination of strictly dominated strategies
* If n is finite and for all i, S_i is finite, exists mixed NE

## The Two-Step

n repeated game, n repeated NE

[Zero-sum games](http://www.autonlab.org/tutorials/gametheory.html)
[Non-zero-sum games](http://www.autonlab.org/tutorials/nonzerosum.html)

Prisoner's dilemma, change game so that both players care about each other