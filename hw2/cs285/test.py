import numpy as np
T = 4
for t in np.arange(T):
    to_go = list[np.arange(t,T)]
    print(to_go)
q_value_to_go = []
gamma = 0.9
rewards = [1,2,1,2]
# for t in np.arange(T):
#     to_go = np.arange(t, T)
#     print(gamma**to_go)

#     print(np.sum((gamma ** to_go) * rewards[to_go]))
#     print(q_value_to_go)
# cumulative_reward = 0
# q_values_to_go = np.zeros(T)

# for t in reversed(range(T)):
#     cumulative_reward = rewards[t] + (gamma * cumulative_reward)
#     q_values_to_go[t] = cumulative_reward

# print(q_values_to_go)

# q_values = []
# T = len(rewards)
# for t, reward in enumerate(rewards):
#     q_value = np.sum([(gamma ** t) * reward])
#     print(type(q_value))
#     q_values.append(q_value)

discounted_sum = 0
for t, reward in enumerate(rewards):
    discounted_sum += (gamma ** t) * reward
print(discounted_sum)
