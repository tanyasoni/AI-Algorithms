# AI-Algorithms
I will be publishing the algorithms I learn while pursuing MSc in Artificial Intelligence.

1) Q Learning
Q Learning is a Reinforcement Learning technique in which an agent learns by exploring the system and receiving positive/negative rewards for it's actions. Every state has an associated reward value for every possible action that can be performed by the agent in that particular state.
              Q(state,action) = Q(state,action) + alpha(reward + gamma(max(next_state,action)) - Q(state,action))
alpha -> Learning Rate
gamma -> Discount Factor (Balances future vs immediate reward)
See the Grid world at grid.png
