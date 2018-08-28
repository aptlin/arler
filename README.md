# Arler: Toy Explorer of Hierarchical Reinforcement Learning

> This repo is under active development. If you find any slip-ups or want to contribute, please [let me know](mailto:sasha@sdll.space).

## Introduction

Learning through play is a simple and powerful form of making sense of the world. Whether animals or machines, learning agents source information and engage with the environment, both of which can inflict significant costs.

Resource scarcity calls for organised conjuring of effective policies that map situations to actions. Some domains offer obvious quantifiable metrics of successful interaction: scores in games, grades in schools, wages and profits in companies. _Reinforcement learning_ gives a range of methods to maximise the numerical reward.

In a classical setting, no prior knowledge is available to the learner, who must discover optimal policies on their own by trial and error, with immediate or delayed rewards.

The sheer range of options available for discovery is often overwhelming, which mandates _transfer learning_: breaking the problem down into essential components and training the agent in a simpler context, with a promise of cheaper exploration and exploitation of the underlying structure.

Invariants of multiple domains might make planning for optimal engagement more efficient, helping us build a hierarchy of routines that narrow down immediate view and speed up the process of learning.

The primary focus of this project is to find and engage in optimal subroutines automatically.

Following Neville Mehta _et al_ (see [[1]](https://ir.library.oregonstate.edu/downloads/6395wb334), [[2]](https://pdfs.semanticscholar.org/cb3a/ef8917900cea4492899bd1f724bcb5e98b8f.pdf)), we make three assumptions:

1. Any sequence of actions that achieves the target goal has a well-defined reason that explains its success.
2. Hints of causality uncover an optimal task hierarchy.
3. Hierarchies are more transferrable across domains than flat sequences of actions.

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

Task hierarchies need a tractable and lightweight representation, and the MAXQ framework provides one of them (see [[4]](https://pdfs.semanticscholar.org/fdc7/c1e10d935e4b648a32938f13368906864ab3.pdf) for details).

In the MAXQ framework, the task-subtask relationships are represented as a directed acyclic graph. Leaves of this task graph correspond to primitive actions, while internal nodes correspond to composite tasks, each of which has the termination predicate describing what goal to achieve and what conditions it needs to succeed, as well as the set of environment variables to track.

### Causal Analysis & Relevance Annotation

Activity models give the opportunity to add information to a successful trajectory of states and activities which then the algorithm can use to build a hierarchy. Thus, for each action we can identify _relevant_ agent and environment variables by considering whether the execution of the action requires their testing or changing in the context. You can see whether some variable is tested or changed by considering how the corresponding conditional probability distribution is changed. If the probability that the variable will conserve its value after the action is completed is less than one, then the variable is context-changed. If the variable is checked to get a reward or the action model shows that this variable is related to some other context-changed variable, then this variable is context-tested.

Using this vocabulary, we say that a variable is relevant to some action if it is context-changed or context-tested.

Let's label a given successful trajectory as follows: we add an edge between two actions if some variable is relevant to both but to no actions in between. The direction of the edge is determined by the precedence of one action over another.

Call such edges relevant. The corresponding _relevantly annotated trajectory_ (RAT for short) is a directed graph of actions from the original trajectory as nodes connected by all the relevant edges, with cycles and failed actions removed.

Partitioning a RAT, you will get a hierarchy to feed into MAXQ.

## Algorithm

### Overview

We use the algorithm of hierarchy induction via models and trajectories (HI-MAT) to partition a RAT into candidate subtasks, given DBN action models and the MDP target goal as input.

The algorithm proceeds backwards from the target goal. Processing each subgoal through the lens of action models, it parses preconditions and derives RAT segments of smaller size, merges them conditionally and recurses deeper down. The base case assumes a singleton RAT segment. The algorithm terminates when the RAT is a single action or a collection of action blocks with identical termination conditions and children, in which case resolution of any higher degree is assumed to lead to no new information-rich abstractions.

| Input          | Output                            |
| -------------- | --------------------------------- |
| action models  | Root action of the task hierarchy |
| RAT            |
| goal predicate |

| HI-MAT                                                                                                                                                                                                                                                                                    |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Base Case**                                                                                                                                                                                                                                                                             |
| If the trajectory consists of a single action, then this action is the root node. The only relevant environment variables are the variables relevant to this action.                                                                                                                      |
| If, on the other hand, actions in the trajectory have identical relevance, then bundle them together under the root node with the given goal predicate as termination condition and set relevant variables to the combination of relevant variables from the action models and predicate. |
| If neither of the conditions above holds, partition the RAT into segments by figuring out the largest subgraph containing only the relevant variables inside.                                                                                                                             |
| If some segment coincides with the entire trajectory, split it in two, separating the ultimate action from the rest.                                                                                                                                                                      |
| Merge all overlapping segments into one.                                                                                                                                                                                                                                                  |
| Invoke the algorithm recursively on each segment, and add the result to the returned root node.                                                                                                                                                                                           |
| Set the termination condition for the root node to the goal predicate.                                                                                                                                                                                                                    |
| Merge models of the actions present in the trajectory to make the composite model of the returned task.                                                                                                                                                                                   |
| Add every primitive action with the model isomorphic to the subgraph of the composite model built in the previous step as a child of the returned task.                                                                                                                                   |
| Return the root node.                                                                                                                                                                                                                                                                     |

### Task Discovery

The target predicate gives the initial set of variables to consider: the _goal set_. The algorithm looks at each literal inside and follows corresponding edges to determine segment boundaries using a simple iterative rule.

If this segment is neither the state it has started with nor the entire trajectory, all the literals entering the segment are added to the goal set. We do this check in order to prevent redundancy.

If, however, it coincides with the entire trajectory, the ultimate action affects only one literal, so the algorithm splits the RAT in two: the first part includes parents of the ultimate action together with its preconditions (which determine the goal predicate of the segment), while the second contains only the ultimate action.

The algorithm then goes on scanning until it accounts for all of the subgoal relevant variables, generating structured batches of sequential actions to feed into the MAXQ learning procedure.

### Specialisation & Termination

A set of tasks combined with the termination condition define a composite task. The subtasks are determined by the next recursive call of the algorithm, while the termination condition is built from the predicate corresponding to the segment and the action model by picking out matching variables.

### Abstraction

The algorithm attempts to find the smallest number of relevant environment variables corresponding to each task in order to speed up the learning process. The heuristic used to achieve this is as follows.

First, we construct a composite action model by merging the DBNs of primitive actions contained in the corresponding RAT segment. This gives us the composite DBN of any task that combines these primitives.

Secondly, we take the union of the set of relevant variables associated with the merged DBN and the set of variables comprising the termination predicate.

Finally, if some of the variables involved in the relational termination condition of the task were left out, we add them as well. This, in effect, parametrises the task and makes it dependent on the context encoded in the current state.

### Generalisation

Since the algorithm works with a single successful trajectory which might encode only a limited amount of information about the environment, in order to maximise the quality of transfer it verifies that all the useful primitive actions have been incorporated in the resultant hierarchy.

The utility of a primitive action not in view is decided by checking whether its DBN is a subgraph of the merged DBN associated with primitive actions already in use. This heuristic is based on the assumption that unobserved primitives with familiar structures achieve the same goal as the primitive children of the task.

## Empirical Evaluation

## Conclusion

Suppose that our agent has simultaneous objectives. A taxi cab, for example, should pick up, move to the proper destination and drop off the passenger.

Some combination of primary actions (go north, west, south, east, pick up, drop off) will achieve this.

Such tasks target a known conjunctive goal: there is a non-empty set of goals states, and the agent must reach them in the shortest period of time.

## References

1. Mehta, Neville. ["Hierarchical Structure Discovery and Transfer in Sequential Decision Problems"](https://ir.library.oregonstate.edu/downloads/6395wb334) (2011)

2. Mehta, Neville; Ray, Soumya; Tadepalli, Prasad; Dietterich, Thomas. ["Automatic discovery and transfer of task hierarchies in reinforcement learning"](https://pdfs.semanticscholar.org/cb3a/ef8917900cea4492899bd1f724bcb5e98b8f.pdf) (2011)

3. Sutton, Richard S.; Barto, Andrew G. ["Reinforcement Learning: An Introduction"](http://incompleteideas.net/book/bookdraft2017nov5.pdf) (2017)

4. Dietterich, Thomas G. ["The MAXQ Method for Hierarchical Reinforcement Learning"](https://pdfs.semanticscholar.org/fdc7/c1e10d935e4b648a32938f13368906864ab3.pdf) (2000)
