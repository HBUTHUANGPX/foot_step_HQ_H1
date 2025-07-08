# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
from .base_storage import BaseStorage
from learning.utils.utils import split_and_pad_trajectories, unpad_trajectories


class RolloutStorage(BaseStorage):
    """A standard rollout storage, implemented for for PPO."""

    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        num_obs,
        num_critic_obs,
        num_actions,
        device="cpu",
    ):

        self.device = device

        self.num_obs = num_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        # num_transitions_per_env *= 2
        # Core
        self.observations = torch.zeros(
            num_transitions_per_env, num_envs, num_obs, device=self.device
        )
        if num_critic_obs is not None:
            self.critic_observations = torch.zeros(
                num_transitions_per_env, num_envs, num_critic_obs, device=self.device
            )
        else:
            self.critic_observations = None

        self.rewards = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.actions = torch.zeros(
            num_transitions_per_env, num_envs, num_actions, device=self.device
        )
        self.dones = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        ).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.values = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.returns = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.advantages = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.mu = torch.zeros(
            num_transitions_per_env, num_envs, num_actions, device=self.device
        )
        self.sigma = torch.zeros(
            num_transitions_per_env, num_envs, num_actions, device=self.device
        )

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.fill_count = 0

    def add_transitions(self, transition: Transition):
        if self.fill_count >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.fill_count].copy_(transition.observations)
        if self.critic_observations is not None:
            self.critic_observations[self.fill_count].copy_(
                transition.critic_observations
            )
        self.actions[self.fill_count].copy_(transition.actions)
        self.rewards[self.fill_count].copy_(transition.rewards.view(-1, 1))
        self.dones[self.fill_count].copy_(transition.dones.view(-1, 1))
        self.values[self.fill_count].copy_(transition.values)
        self.actions_log_prob[self.fill_count].copy_(
            transition.actions_log_prob.view(-1, 1)
        )
        self.mu[self.fill_count].copy_(transition.action_mean)
        self.sigma[self.fill_count].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        self.fill_count += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = (
            hidden_states[0]
            if isinstance(hidden_states[0], tuple)
            else (hidden_states[0],)
        )
        hid_c = (
            hidden_states[1]
            if isinstance(hidden_states[1], tuple)
            else (hidden_states[1],)
        )

        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(
                    self.observations.shape[0], *hid_a[i].shape, device=self.device
                )
                for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(
                    self.observations.shape[0], *hid_c[i].shape, device=self.device
                )
                for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.fill_count].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.fill_count].copy_(hid_c[i])

    def clear(self):
        self.fill_count = 0

    # def compute_returns(self, last_values, gamma, lam):
    #     num_transitions_per_env = self.num_transitions_per_env // 2
    #     advantage = 0
    #     resize = lambda x: x.view(num_transitions_per_env, 2, -1, 1)
    #     for step in reversed(range(num_transitions_per_env)):
    #         if step == num_transitions_per_env - 1:
    #             next_values = last_values
    #         else:
    #             next_values = self.values[step + 1]
    #         next_is_not_terminal = 1.0 - resize(self.dones)[step].float()
    #         delta = (
    #             resize(self.rewards)[step]
    #             + next_is_not_terminal * gamma * next_values
    #             - resize(self.values)[step]
    #         )
    #         advantage = delta + next_is_not_terminal * gamma * lam * advantage
    #         resize(self.returns)[step] = advantage + resize(self.values)[step]

    #     # Compute and normalize the advantages
    #     self.advantages = self.returns - self.values
    #     self.advantages = (self.advantages - self.advantages.mean()) / (
    #         self.advantages.std() + 1e-8
    #     )

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = (
                self.rewards[step]
                + next_is_not_terminal * gamma * next_values
                - self.values[step]
            )
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (
                flat_dones.new_tensor([-1], dtype=torch.int64),
                flat_dones.nonzero(as_tuple=False)[:, 0],
            )
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size, requires_grad=False, device=self.device
        )

        observations = self.observations.flatten(0, 1)
        if self.critic_observations is not None:
            critic_observations = self.critic_observations.flatten(0, 1)
        else:
            critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                yield obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    None,
                    None,
                ), None

    # for RNNs only
    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        # print(self.observations.shape,self.dones.shape)
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(
            self.observations, self.dones
        )

        if self.critic_observations is not None:
            padded_critic_obs_trajectories, _ = split_and_pad_trajectories(
                self.critic_observations, self.dones
            )
        else:
            padded_critic_obs_trajectories = padded_obs_trajectories

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[
                    :, first_traj:last_traj
                ]

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][
                        first_traj:last_traj
                    ]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][
                        first_traj:last_traj
                    ]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c
                ]
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_a_batch

                yield obs_batch, critic_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    hid_a_batch,
                    hid_c_batch,
                ), masks_batch

                first_traj = last_traj

    def obs_symmetry(self, tensor, critic_flag=False):
        # 构造对角矩阵M
        # n = tensor.shape[-1]  # 最后一维大小
        # device = tensor.device
        # diag_elements = torch.ones(n, device=device)
        # indices_to_negate = [
        #     0,2,
        #     4,
        #     7,8,
        #     9,10,
        #     11,13,16,17,19,22,
        #     23,25,28,29,31,34,
        #     36,38,40,42,
        # ]
        # if critic_flag:
        #     indices_to_negate.extend([46,49,51,53,55])  # 根据critic_flag添加
        # # print(indices_to_negate)
        # # indices_to_negate = [i for i in indices_to_negate if i < n]
        # # print(indices_to_negate)
        # diag_elements[indices_to_negate] = -1
        # M = torch.diag(diag_elements)
        # # 构造置换矩阵P
        # P = torch.eye(n, device=device)
        # # 示例：交换dof_pos的索引（根据你的代码）
        # swap_pairs = [(11,17),(12,18),(13,19),(14,20),(15,21),(16,22),
        #               (23,29),(24,30),(25,31),(26,32),(27,33),(28,34),
        #               (35,39),(36,40),(37,41),(38,42),
        #               ]
        # if critic_flag:
        #     swap_pairs.extend([(48,52),(49,53),(50,54),(51,55)]) 
        # # print(swap_pairs)
        # # swap_pairs = [(i, j) for i, j in swap_pairs if i < n and j < n]
        # # print(swap_pairs)
        # for i, j in swap_pairs:
        #     P[[i, j]] = P[[j, i]]
        # # print(len(swap_pairs),P.size(),M.size(),tensor.size())
        # s_tensor = tensor @ P @ M

        # if critic_flag:
        #     s_tensor = tensor @ self.critic_P @ self.critic_M
        # else:
        #     s_tensor = tensor @ self.actor_P @ self.actor_M

        s_tensor = torch.zeros_like(tensor)
        a = 0
        # # base heading
        # s_tensor[..., a] = -tensor[..., a]
        # a += 1
        # base ang vel
        s_tensor[..., a] = -tensor[..., a] # 0
        s_tensor[..., a + 1] = tensor[..., a + 1]
        s_tensor[..., a + 2] = -tensor[..., a + 2]
        a += 3
        # projected gravity
        s_tensor[..., a] = tensor[..., a]#3
        s_tensor[..., a + 1] = -tensor[..., a + 1]
        s_tensor[..., a + 2] = tensor[..., a + 2]
        a += 3
        # commands
        s_tensor[..., a] = tensor[..., a]#6
        s_tensor[..., a + 1] = -tensor[..., a + 1]
        s_tensor[..., a + 2] = -tensor[..., a + 2]
        a += 3
        # # phase sin/cos
        s_tensor[..., a] = -tensor[..., a]#9
        s_tensor[..., a + 1] = -tensor[..., a + 1]
        a += 2
        # dof pos
        s_tensor[..., a] = -tensor[..., a + 6]#11
        s_tensor[..., a + 1] = tensor[..., a + 7]
        s_tensor[..., a + 2] = -tensor[..., a + 8]
        s_tensor[..., a + 3] = tensor[..., a + 9]
        s_tensor[..., a + 4] = tensor[..., a + 10]
        s_tensor[..., a + 5] = -tensor[..., a + 11]

        s_tensor[..., a + 6] = -tensor[..., a + 0]#17
        s_tensor[..., a + 7] = tensor[..., a + 1]
        s_tensor[..., a + 8] = -tensor[..., a + 2]
        s_tensor[..., a + 9] = tensor[..., a + 3]
        s_tensor[..., a + 10] = tensor[..., a + 4]
        s_tensor[..., a + 11] = -tensor[..., a + 5]
        a += 12
        # dof vel
        s_tensor[..., a] = -tensor[..., a + 6]#23
        s_tensor[..., a + 1] = tensor[..., a + 7]
        s_tensor[..., a + 2] = -tensor[..., a + 8]
        s_tensor[..., a + 3] = tensor[..., a + 9]
        s_tensor[..., a + 4] = tensor[..., a + 10]
        s_tensor[..., a + 5] = -tensor[..., a + 11]

        s_tensor[..., a + 6] = -tensor[..., a]
        s_tensor[..., a + 7] = tensor[..., a + 1]
        s_tensor[..., a + 8] = -tensor[..., a + 2]
        s_tensor[..., a + 9] = tensor[..., a + 3]
        s_tensor[..., a + 10] = tensor[..., a + 4]
        s_tensor[..., a + 11] = -tensor[..., a + 5]
        a += 12
        # foot_states_right & foot_states_left
        s_tensor[..., a] = tensor[..., a + 4]#35
        s_tensor[..., a + 1] = -tensor[..., a + 5]
        s_tensor[..., a + 2] = tensor[..., a + 6]
        s_tensor[..., a + 3] = -tensor[..., a + 7]

        s_tensor[..., a + 4] = tensor[..., a]
        s_tensor[..., a + 5] = -tensor[..., a + 1]
        s_tensor[..., a + 6] = tensor[..., a + 2]
        s_tensor[..., a + 7] = -tensor[..., a + 3]
        a += 8
        # standing_command_mask
        s_tensor[..., a] = tensor[..., a] # 43
        a += 1

        if critic_flag:
            # base_height
            s_tensor[..., a] = tensor[..., a] #44
            a += 1
            # base_lin_vel_world
            s_tensor[..., a] = tensor[..., a] #45
            s_tensor[..., a + 1] = -tensor[..., a + 1]
            s_tensor[..., a + 2] = tensor[..., a + 2]
            a += 3
            # # step_commands_right
            # s_tensor[..., a] = tensor[..., a + 4] #48
            # s_tensor[..., a + 1] = -tensor[..., a + 5]
            # s_tensor[..., a + 2] = tensor[..., a + 6]
            # s_tensor[..., a + 3] = -tensor[..., a + 7]
            # s_tensor[..., a + 4] = tensor[..., a]
            # s_tensor[..., a + 5] = -tensor[..., a + 1]
            # s_tensor[..., a + 6] = tensor[..., a + 2]
            # s_tensor[..., a + 7] = -tensor[..., a + 3]
            # a += 8
        return s_tensor

    def action_symmetry(self, tensor):
        s_tensor = torch.zeros_like(tensor)
        a = 0
        s_tensor[..., a] = -tensor[..., a + 6]
        s_tensor[..., a + 1] = tensor[..., a + 7]
        s_tensor[..., a + 2] = -tensor[..., a + 8]
        s_tensor[..., a + 3] = tensor[..., a + 9]
        s_tensor[..., a + 4] = tensor[..., a + 10]
        s_tensor[..., a + 5] = -tensor[..., a + 11]

        s_tensor[..., a + 6] = -tensor[..., a]
        s_tensor[..., a + 7] = tensor[..., a + 1]
        s_tensor[..., a + 8] = -tensor[..., a + 2]
        s_tensor[..., a + 9] = tensor[..., a + 3]
        s_tensor[..., a + 10] = tensor[..., a + 4]
        s_tensor[..., a + 11] = -tensor[..., a + 5]
        return s_tensor
        # return tensor @ self.action_P @ self.action_M
