# AI-Algorithms
I will be publishing the algorithms I learn while pursuing MSc in Artificial Intelligence.

1) Q Learning for Taxi problem (Dietterich,2000)
Q Learning is a Reinforcement Learning technique in which an agent learns by exploring the system and receiving positive/negative rewards for it's actions. Every state has an associated reward value for every possible action that can be performed by the agent in that particular state.
    Q(state,action) = Q(state,action) + alpha*(reward + gamma*(max(Q(next_state,action))) - Q(state,action))
alpha -> Learning Rate
gamma -> Discount Factor (Balances future vs immediate reward)
The algorithm uses an exploration strategy 10% of the time.
See the Grid world at grid.png

2) SARSA - State-Action-Reward-State-Action for Taxi problem (Dietterich,2000)
SARSA algorithm is similar to the Q Learning algorithm, except that it is an On Policy reinforcement learning algorithm. In Q Learning we assume that the most optimal policy is being followed. But in SARSA we update the Q value of a state based on the true action value that we are going to take in the next state.
    Q(state,action) = Q(state,action) + alpha*(reward + gamma*(Q(next_state,next_action)) - Q(state,action))
