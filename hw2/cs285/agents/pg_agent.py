from typing import Optional, Sequence
import numpy as np
import torch

from cs285.networks.policies import MLPPolicyPG
from cs285.networks.critics import ValueCritic
from cs285.infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network # learn policy
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed # learn value function or Q, given state or obs, V(s) or Q(a,s)
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards) # a list of 5 arrays (1000 elements each)
        # print('q_values', type(q_values), len(q_values[0]))
        # q_values is a list of arrays, where each array corresponds to the Q-values for a single trajectory.

        # TODO: flatten the lists of arrays into single arrays, so that the rest of the code can be written in a vectorized
        # way. obs, actions, rewards, terminals, and q_values should all be arrays with a leading dimension of `batch_size`
        # beyond this point.
        # Flattened terminals: [0 0 1 0 1 0 0 0 1]
        # print('obs shape', len(obs), obs[0].shape) # a list of 43 trajectories)
        # each list has at least 1000 elements (the minimum timestep per batch)

        # obs = np.array(obs)
        # actions = np.array(actions)
        # rewards = np.array(rewards)
        # terminals = np.array(terminals)
        # q_values = np.array(q_values)
        obs = np.concatenate(obs) 
        actions = np.concatenate(actions)
        rewards = np.concatenate(rewards)
        terminals = np.concatenate(terminals)
        q_values = np.concatenate(q_values)
        # print('q values', q_values, type(q_values), len(q_values))
        # <class 'numpy.ndarray'> 5000
        # print(obs.shape, actions.shape, rewards.shape, terminals.shape, len(q_values))


        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(
            obs, rewards, q_values, terminals
        )

        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        # TODO: update the PG actor/policy network once using the advantages
        info: dict = self.actor.update(obs, actions, advantages)

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        if self.critic is not None:
            # TODO: perform `self.baseline_gradient_steps` updates to the critic/baseline network
            critic_info: dict = self.critic.update(obs, q_values)
            info.update(critic_info)

        return info

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""
        q_values = []
        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            # TODO: use the helper function self._discounted_return to calculate the Q-values
            for reward in rewards:
                # print(reward, type(reward)) # <class 'numpy.ndarray'>
                q_value_arr = self._discounted_return(reward)
                q_values.append(q_value_arr)
        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            # TODO: use the helper function self._discounted_reward_to_go to calculate the Q-values
            for reward in rewards:
                q_value_arr = self._discounted_reward_to_go(reward)
                q_values.append(q_value_arr)

        return q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        if self.critic is None:
            # TODO: if no baseline, then what are the advantages?
            advantages = q_values
        else:
            # TODO: run the critic and use it as a baseline
            # print(type(obs), obs, obs.shape) # <class 'numpy.ndarray'>, (5000, 17)
            obs = ptu.from_numpy(obs)
            values = self.critic(obs)
            values = np.array(values.cpu().detach().numpy())
            # print(type(q_values), len(q_values))
            assert values.shape == q_values.shape

            if self.gae_lambda is None:
                # TODO: if using a baseline, but not GAE, what are the advantages?
                advantages = q_values - values
            else:
                # TODO: implement GAE
                batch_size = obs.shape[0]

                # HINT: append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0]) # <class 'numpy.ndarray'>
                advantages = np.zeros(batch_size + 1)
                rewards = np.append(rewards, [0]) # dummy reward C
                q_values = np.append(q_values, [0])
                delta = q_values - values

                for i in reversed(range(batch_size)):
                    # TODO: recursively compute advantage estimates starting from timestep T.
                    # HINT: use terminals to handle edge cases. terminals[i] is 1 if the state is the last in its
                    # trajectory, and 0 otherwise.
                    if terminals[i]:
                        advantages[i] = rewards[i] - values[i]
                    else:
                        advantages[i] = delta[i] + self.gamma * self.gae_lambda * advantages[i+1]

                # remove dummy advantage
                advantages = advantages[:-1]
        # print(advantages, type(advantages), len(advantages))
        # advs = np.concatenate(advantages)
        # TODO: normalize the advantages to have a mean of zero and a standard deviation of one within the batch
        if self.normalize_advantages:
            eps = 1e-8
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + eps)
        # print(type(advantages)) # list
        # print(advs, type(advs), len(advs))
        return advantages

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!

        Specifically, the function should take a list of arrays (where each array corresponds to the rewards 
        from a single trajectory) and return a list of floats, where each float is the discounted return for that trajectory.
        """

        discounted_sum = 0
        for reward in reversed(rewards):
            discounted_sum = self.gamma * discounted_sum + reward
            # print([discounted_sum] * len(rewards))
        return [discounted_sum] * len(rewards)
        # for t, reward in enumerate(rewards):
        #     discounted_sum += (self.gamma ** t) * reward
        #     # print(t, reward, discounted_sum)
        #     discounted_return.append([discounted_sum] * len(rewards))
        #     # print(discounted_return, type(discounted_return), len(discounted_return))
        #     return discounted_return


    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        """
        # T = len(rewards)
        # rewards_to_go = np.zeros(T)
        # cumulative_reward = 0
        # for t in reversed(range(T)):
        #     cumulative_reward = rewards[t] + (self.gamma * cumulative_reward)
        #     rewards_to_go[t] = cumulative_reward

        # return rewards_to_go # a list 

        discounted_return = []  
        discounted_sum = 0
        for reward in reversed(rewards):
            discounted_sum = self.gamma * discounted_sum + reward
            discounted_return.insert(0, discounted_sum)
        return discounted_return

