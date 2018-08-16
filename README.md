# This repo is under active development. If you find any slip-ups or want to contribute, please [let me know](mailto:sasha@sdll.space).

# Hierarchy Induction via Models and Trajectories (HI-MAT)

## Introduction

Learning through play is a primitive and powerful form of making sense of the world. Whether animals or machines, learning agents are capable of sourcing information and engaging with the environment through exploration and exploitation, both of which can inflict significant costs. Universal public education and plastic pollution serve as examples of problems with a plethora of interacting agents and changing environments.

Scarcity of resources calls for a structured mode of building effective policies mapping situations to actions. Some domains offer natural quantifiable metrics of successful interaction: scores in games, grades in schools, wages, profits and impact in companies. _Reinforcement learning_ gives a set of methods to maximise a numerical reward signal. In a classical setting, no prior knowledge is available to the learner, who must discover an optimal policy on their own by trial and error, with immediate or delayed rewards.

Some domains, such as task scheduling or travel planning, present an overwhelming range of options for exploration, and mandate the approach of _transfer learning_: breaking down the problem into essential components and training the agent in a simplified setting, with a promise of faster and cheaper discovery of the underlying structure. One insight worth pursuing is the idea that discovering and representing invariants characteristic of multiple domains will help determine sequences of actions necessary to achieve efficient planning and optimal modes of engagement. Often these sequences can be further organised into a hierarchy of distinct macros, which can then speed up the process of learning. Automating the process of finding optimal subroutines is the focus of this project.

Built upon the work of Neville Mehta _et al_ (see [[1]](https://ir.library.oregonstate.edu/downloads/6395wb334), [[2]](https://pdfs.semanticscholar.org/cb3a/ef8917900cea4492899bd1f724bcb5e98b8f.pdf)), our two key assumptions are:

1. Any successful sequence of actions supplied with the knowledge of causal relationships can help uncover an optimal task hierarchy.
2. Task hierarchies are more transferable across domains than knowledge pertaining to the value of being in a particular state or the utility of each action on its own.

Exploitation of the action models encoding the environmental dynamics allows us to compose a task hierarchy with minimal causal dependencies between distinct tasks. The goal is to produce a meaningful schema to use for any targets with similar causal structures.

The next section describes the technical toolkit used to develop the method. Then we describe the algorithm, benchmark its performance on a toy domain, conclude with the criticism of the general approach and lay out possible improvements.

## Technical Background

Markov decision processes (MDPs) provide the framework necessary to formalise the problem of sequential decision-making. A set of _states_ of the environment, _actions_ available to the agent, the _transition function_ specifying how each action affect the state, and the _reward function_ representing the feedback to the agent completely determine an MDP.

The transition and reward functions can behave stochastically: they can change subject to random variations.

The set of environment states can be _factored_, so that each state has several variables that describe it fully. In case of a helicopter as an agent, for example, its velocity and position can characterise the state of the environment.

Note that MDPs capture the agent-environment interaction in a computationally amicable manner: continual engagement is broken down into discrete time steps, with the agent receiving the information about the environment in the form of a _state_ id, on the basis of which it picks the next action. Each action is assigned a numerical reward, which the agent receives at some point, and the environment transitions to a new state.

We will consider only finite MDPs, so that the sets of all the possible states, actions and rewards are discrete. In this way we can impose the Markov property on the transition function, describing each step and its reward at a particular point in time as random variables with well-defined discrete probability distributions dependent only on the previous step.

Note that this model is completely defined by the probability function, showing how likely the agent is to find itself in the current state and receive some reward based on its previous behaviour summed up by the state of the environment and taken action.

The agent must choose a policy which determines the action it takes in each state. The main problem which the agent must solve is to find the optimal strategy maximising its reward. Since we consider only finite MDPs, the cumulative reward obtained by the agent is finite and fixed by the chosen policy. This allows us to compute the value of each state with respect to the policy: dynamic programming is a classic approach. Alas, it suffers from the curse of dimensionality, breaking down when the number of states is exponential in the number of state variables, which is characteristic of the real-world domains.

One trick to resolve the issue is to divide and conquer: hierarchical reinforcement learning decomposes the original MDPs into more manageable chunks, solves them one by one and reassembles intermediate results into a single policy.

### Action models

Discovery of task hierarchy in our approach relies on action models easy to store and access.

Each action can be described in terms of its effect on the agent and the environment with a _dynamic probability network_, showing how each variable in the current state will influence all the variables in the next, including the effect of each variable on the reward. Conditional probability trees quantify this influence, and describe how likely the assignment of the value is after the action is done. In this way, the Markovian absence of memory comes in handy and allows us to describe action models as bipartite graphs (we assume that no variables affect each other within the action step).

The left half of the graph consists of nodes corresponding to environment variables before the action is executed. The nodes on the right, including a special node for the reward, represent the variables after the action is completed, each containing an inspectable representation of the decision tree, with the respective variables as leaves and parents as internal nodes.

### Task Hierarchies & Goal Predicates

Successful abstraction of primitive tasks into more complex macros with greater time frames can speed up learning and break down intractable planning problems into manageable subtasks.

##### MAXQ

### Causal Analysis & Relevance Annotation

Activity models give the opportunity to add information to a successful trajectory of states and activities which then the algorithm can use to build a hierarchy. Thus, for each action we can identify _relevant_ agent and environment variables by considering whether the execution of the action requires their testing or changing in the context. You can see whether some variable is tested or changed by considering how the corresponding conditional probability distribution is changed. If the probability that the variable will conserve its value after the action is completed is less than one, then the variable is context-changed. If the variable is checked to get a reward or the action model shows that this variable is related to some other context-changed variable, then this variable is context-tested.

Using this vocabulary, we say that a variable is relevant to some action if it is context-changed or context-tested.

Let's label a given successful trajectory as follows: we add an edge between two actions if some variable is relevant to both but to no actions in between. The direction of the edge is determined by the precedence of one action over another.

Call such edges relevant. The corresponding _relevantly annotated trajectory_ (RAT for short) is a directed graph of actions from the original trajectory as nodes connected by all the relevant edges, with cycles and failed actions removed.

Partitioning a RAT, you will get a hierarchy to feed into MAXQ.

## Algorithm

### Overview

### Task Discovery

### Specialisation & Termination

### Abstraction

### Generalisation

## Empirical Evaluation

## Conclusion

Suppose that our agent has simultaneous objectives. A taxi cab, for example, should pick up, move to the proper destination and drop off the passenger.

Some combination of primary actions (go north, west, south, east, pick up, drop off) will achieve this.

Such tasks target a known conjunctive goal: there is a non-empty set of goals states, and the agent must reach them in the shortest period of time.

Input:

- action models
- RAT
- goal predicate

Output:

- RAT partition to be fed into MAXQ

## References

1. Mehta, Neville. ["Hierarchical Structure Discovery and Transfer in Sequential Decision Problems"](https://ir.library.oregonstate.edu/downloads/6395wb334) (2011)

2. Mehta, Neville; Ray, Soumya; Tadepalli, Prasad; Dietterich, Thomas. ["Automatic discovery and transfer of task hierarchies in reinforcement learning"](https://pdfs.semanticscholar.org/cb3a/ef8917900cea4492899bd1f724bcb5e98b8f.pdf) (2011)

3. Sutton, Richard S.; Barto, Andrew G. ["Reinforcement Learning: An Introduction"](http://incompleteideas.net/book/bookdraft2017nov5.pdf) (2017)
