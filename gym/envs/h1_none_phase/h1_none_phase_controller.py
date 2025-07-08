"""
Hierarchical structure for Deep Stepper for Humanoid
1) Low-level policy: Step controller trained by PPO
    - It is divided into two section. (1) Only one-step controller (2) Continuous-step controller
2) High-level policy: Step planner trained by SAC

Purpose: Given a base velocity command (linear x,y velocity, angular velocity), 
         robot determines stepping locations to follow the commanded velocity

This script serves as a Low-level policy which actuate the robot to take a step

* All variables are calculated w.r.t world frame
* However, when the variables are put into observation, it is converted w.r.t base frame
"""

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from gym.envs.h1_none_phase.h1_none_phase_controller_config import H1_NPControllerCfg
from gym.utils.math import *
from gym.envs import LeggedRobot
from isaacgym import gymapi, gymutil
import numpy as np
from typing import Tuple, Dict
from .h1_utils import (
    FootStepGeometry,
    SimpleLineGeometry,
    VelCommandGeometry,
    smart_sort,
)
from gym.utils import XCoMKeyboardInterface
from .jacobian import apply_coupling
from scipy.signal import correlate
import torch.nn.functional as F


@torch.jit.script
def torch_rand_float_2(lower, midupper, midlower, upper, shape, device):
    # type: (float, float, float, float, Tuple[int, int], str) -> Tensor
    """
    生成在两个不连续区间内的随机浮点数:
    [lower, midupper] 和 [midlower, upper]

    参数:
        lower: 第一个区间的下限
        upper: 第二个区间的上限
        midupper: 第一个区间的上限 (必须小于 midlower)
        midlower: 第二个区间的下限 (必须大于 midupper)
        shape: 输出张量的形状
        device: 计算设备

    返回:
        指定形状的张量，其元素值位于 [lower, midupper] ∪ [midlower, upper]
    """
    # 验证输入参数的有效性
    assert midupper < midlower, "midupper must be less than midlower"

    # 计算两个区间的长度
    range1 = midupper - lower
    range2 = upper - midlower

    # 生成随机数
    rand_vals = torch.rand(*shape, device=device)

    # 将随机数映射到两个区间
    # 首先将随机数缩放到总范围 [0, range1 + range2]
    scaled = rand_vals * (range1 + range2)

    # 然后根据值的大小决定分配到哪个区间
    result = torch.where(
        scaled < range1, scaled + lower, scaled - range1 + midlower  # 第一个区间
    )  # 第二个区间

    return result


class H1_NPController(LeggedRobot):
    cfg: H1_NPControllerCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _setup_keyboard_interface(self):
        self.keyboard_interface = XCoMKeyboardInterface(self)

    def _init_buffers(self):
        super()._init_buffers()
        # * Robot states
        self.base_height = self.root_states[:, 2:3]
        self.right_hip_pos = self.rigid_body_state[
            :, self.rigid_body_idx["right_hip_yaw_link"], :3
        ]
        self.left_hip_pos = self.rigid_body_state[
            :, self.rigid_body_idx["left_hip_yaw_link"], :3
        ]
        self.CoM = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.foot_states = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            7,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # num_envs x (right & left foot) x (x, y, z, quat)
        self.foot_states_right = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )  # num_envs x (x, y, z, heading, projected_gravity)
        self.foot_states_left = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )  # num_envs x (x, y, z, heading, projected_gravity)
        self.foot_heading = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # num_envs x (right & left foot heading)
        self.foot_projected_gravity = torch.stack(
            (self.gravity_vec, self.gravity_vec), dim=1
        )  # (num_envs x 2 x 3), [0., 0., -1.]
        self.foot_contact = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.foot_contact_sequence = torch.zeros(
            self.num_envs,
            len(self.feet_ids),  # 0.02*17=0.34
            (self.cfg.commands.ranges.sample_period[1] * 2 + 1) * 3,
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.last_foot_contact = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.foot_air_time = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.ankle_vel_history = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            2 * 3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.base_heading = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.base_lin_vel_world = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )

        # * Others
        self.update_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
        )  # Number of transition since the beginning of the episode
        self.update_commands_ids = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )  # envs whose step commands are updated
        self.phase_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
        )  # Number of phase progress
        self.update_phase_ids = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )  # envs whose phases are updated
        self.phase = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )  # phase of current step in a whole gait cycle
        self.ICP = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )  # Instantaneous Capture Point (ICP) for the robot
        self.raibert_heuristic = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # step_location & angle by raibert heuristic
        self.w = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )  # eigenfrequency of the inverted pendulum
        self.step_length = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )  # step length
        self.step_width = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )  # step width
        self.dstep_length = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )  # desired step length
        self.dstep_width = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )  # desired step width
        self.support_foot_pos = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )  # position of the support foot
        self.prev_support_foot_pos = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )  # position of the support foot
        self.LIPM_CoM = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )  # base position of the Linear Inverted Pendulum model

        # * Observation variables
        self.phase_sin = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.phase_cos = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.contact_schedule = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.standing_command_mask = torch.zeros(
            self.num_envs, 1, dtype=torch.int64, device=self.device, requires_grad=False
        )
        self.command_catrgories = torch.randint(
            0,
            5,
            (self.num_envs, 1),
            device=self.device,
            requires_grad=False,
        ).squeeze(1)

        self.step_period = torch.zeros(
            self.num_envs, 1, dtype=torch.long, device=self.device, requires_grad=False
        )
        self.full_step_period = torch.zeros(
            self.num_envs, 1, dtype=torch.long, device=self.device, requires_grad=False
        )  # full_step_period = 2 * step_period
        #
        self.feet_air_time = torch.zeros(
            self.num_envs,
            self.feet_ids.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.quadrilateral_point = torch.zeros(
            self.num_envs,
            4,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.step_pos = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.contact_queue = torch.zeros(
            self.num_envs,
            (self.cfg.commands.ranges.sample_period[0] + 1),
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.termination_levels = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.term_base_lin_vel_limit = self.cfg.commands.term_ranges.term_base_lin_vel[
            0
        ] * torch.ones(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.term_base_ang_vel_limit = self.cfg.commands.term_ranges.term_base_ang_vel[
            0
        ] * torch.ones(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.term_projected_gravity_x_limit = (
            self.cfg.commands.term_ranges.term_projected_gravity_x[0]
            * torch.ones(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
        )
        self.term_projected_gravity_y_limit = (
            self.cfg.commands.term_ranges.term_projected_gravity_y[0]
            * torch.ones(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
        )
        self.term_base_pos_limit = self.cfg.commands.term_ranges.term_base_pos[
            0
        ] * torch.ones(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.term_base_lin_vel_step = (
            self.cfg.commands.term_ranges.term_base_lin_vel[1]
            - self.cfg.commands.term_ranges.term_base_lin_vel[0]
        ) / self.cfg.commands.term_ranges.max_level
        self.term_base_ang_vel_step = (
            self.cfg.commands.term_ranges.term_base_ang_vel[1]
            - self.cfg.commands.term_ranges.term_base_ang_vel[0]
        ) / self.cfg.commands.term_ranges.max_level
        self.term_projected_gravity_x_step = (
            self.cfg.commands.term_ranges.term_projected_gravity_x[1]
            - self.cfg.commands.term_ranges.term_projected_gravity_x[0]
        ) / self.cfg.commands.term_ranges.max_level
        self.term_projected_gravity_y_step = (
            self.cfg.commands.term_ranges.term_projected_gravity_y[1]
            - self.cfg.commands.term_ranges.term_projected_gravity_y[0]
        ) / self.cfg.commands.term_ranges.max_level
        self.term_base_pos_step = (
            self.cfg.commands.term_ranges.term_base_pos[1]
            - self.cfg.commands.term_ranges.term_base_pos[0]
        ) / self.cfg.commands.term_ranges.max_level

    def _compute_torques(self):
        self.desired_pos_target = (
            self.dof_pos_target * self.cfg.env.action_scale + self.default_dof_pos
        )
        q = self.dof_pos.clone()
        qd = self.dof_vel.clone()
        q_des = self.desired_pos_target.clone()
        qd_des = torch.zeros_like(self.dof_pos_target)
        tau_ff = torch.zeros_like(self.dof_pos_target)
        kp = self.p_gains.clone()
        kd = self.d_gains.clone()

        torques = kp * (q_des - q) + kd * (qd_des - qd) + tau_ff
        # print("q: ",q[0,:])

        torques = torch.clip(torques, -self.torque_limits, self.torque_limits)

        return torques.view(self.torques.shape)

    def _resample_commands(self, env_ids):
        """Randomly select foot step commands one/two steps ahead"""
        self.command_catrgories[env_ids] = torch.randint(
            0,
            5,
            (len(env_ids), 1),
            device=self.device,
            requires_grad=False,
        ).squeeze(1)
        # 0 is standing，vel x is 0, vel y is 0, ang vel yaw is 0
        # 1 is walking in sagittal, vel y is 0, ang vel yaw is 0
        # 2 is walking laterally, vel x is 0, ang vel yaw is 0
        # 3 is rotating in place, vel x is 0, vel y is 0
        # 4 is omnidirectional walking, all commands are random
        self.commands[env_ids, 0] = torch_rand_float_2(
            self.command_ranges["lin_vel_x"][0],
            -0.2,
            0.2,
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float_2(
            -self.command_ranges["lin_vel_y"],
            -0.2,
            0.2,
            self.command_ranges["lin_vel_y"],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float_2(
            -self.command_ranges["yaw_vel"],
            -0.2,
            0.2,
            self.command_ranges["yaw_vel"],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        if torch.any(self.command_catrgories == 1):
            self.commands[self.command_catrgories == 1, 1] = 0
            self.commands[self.command_catrgories == 1, 2] = 0
            self.standing_command_mask[self.command_catrgories == 1] = 0
        if torch.any(self.command_catrgories == 2):
            self.commands[self.command_catrgories == 2, 0] = 0
            self.commands[self.command_catrgories == 2, 2] = 0
            self.standing_command_mask[self.command_catrgories == 2] = 0
        if torch.any(self.command_catrgories == 3):
            self.commands[self.command_catrgories == 3, 0] = 0
            self.commands[self.command_catrgories == 3, 1] = 0
            self.standing_command_mask[self.command_catrgories == 3] = 0
        if torch.any(self.command_catrgories == 0):
            self.commands[self.command_catrgories == 0, 0] = 0
            self.commands[self.command_catrgories == 0, 1] = 0
            self.commands[self.command_catrgories == 0, 2] = 0
            self.standing_command_mask[self.command_catrgories == 0] = 1

        self.step_period[env_ids] = torch.randint(
            low=self.command_ranges["sample_period"][0],
            high=self.command_ranges["sample_period"][1],
            size=(len(env_ids), 1),
            device=self.device,
        )
        self.full_step_period = 2 * self.step_period

        # * Randomly select the desired step width
        self.dstep_width[env_ids] = torch_rand_float(
            self.command_ranges["dstep_width"][0],
            self.command_ranges["dstep_width"][1],
            (len(env_ids), 1),
            self.device,
        )

    def _reset_system(self, env_ids):
        super()._reset_system(env_ids)
        # * Robot states
        self.foot_states[env_ids] = self._calculate_foot_states(
            self.rigid_body_state[:, self.feet_ids, :7]
        )[env_ids]
        self.foot_projected_gravity[env_ids, 0] = self.gravity_vec[env_ids]
        self.foot_projected_gravity[env_ids, 1] = self.gravity_vec[env_ids]

        # * Others
        self.update_count[env_ids] = 0
        self.update_commands_ids[env_ids] = False
        self.phase_count[env_ids] = 0
        self.update_phase_ids[env_ids] = False
        self.phase[env_ids] = 0
        self.contact_queue[env_ids, ...] = 0
        #
        self.feet_air_time[env_ids] = 0.0

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()

        self._update_robot_states()
        self._calculate_CoM()
        # self._log_info()

    def _update_robot_states(self):
        """Update robot state variables"""
        self.base_height[:] = self.root_states[:, 2:3]
        forward = quat_apply(self.base_quat, self.forward_vec)
        self.base_heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
        self.right_hip_pos = self.rigid_body_state[
            :, self.rigid_body_idx["right_hip_yaw_link"], :3
        ]
        self.left_hip_pos = self.rigid_body_state[
            :, self.rigid_body_idx["left_hip_yaw_link"], :3
        ]
        self.foot_states = self._calculate_foot_states(
            self.rigid_body_state[:, self.feet_ids, :7]
        )

        right_foot_forward = quat_apply(self.foot_states[:, 0, 3:7], self.forward_vec)
        left_foot_forward = quat_apply(self.foot_states[:, 1, 3:7], self.forward_vec)
        right_foot_heading = wrap_to_pi(
            torch.atan2(right_foot_forward[:, 1], right_foot_forward[:, 0])
        )
        left_foot_heading = wrap_to_pi(
            torch.atan2(left_foot_forward[:, 1], left_foot_forward[:, 0])
        )
        self.foot_heading[:, 0] = right_foot_heading
        self.foot_heading[:, 1] = left_foot_heading

        self.foot_projected_gravity[:, 0] = quat_rotate_inverse(
            self.foot_states[:, 0, 3:7], self.gravity_vec
        )
        self.foot_projected_gravity[:, 1] = quat_rotate_inverse(
            self.foot_states[:, 1, 3:7], self.gravity_vec
        )

        self.update_count += 1
        self.phase_count += 1
        self.phase += (1 / self.full_step_period) * 2.0 * torch.pi

        # * Ground truth foot contact
        self.last_foot_contact = self.foot_contact.clone()
        self.foot_contact = torch.gt(self.contact_forces[:, self.feet_ids, 2], 0)
        self.foot_contact_sequence = torch.roll(
            self.foot_contact_sequence, shifts=1, dims=-1
        )
        # 将最新的 foot_contact 插入到最前方（索引 0）
        self.foot_contact_sequence[:, :, 0] = self.foot_contact
        # * Phase-based foot contact
        self.contact_schedule = self.smooth_sqr_wave(
            self.phase * (1 - self.standing_command_mask)
        )

        naxis = 3
        self.ankle_vel_history[:, 0, naxis:] = self.ankle_vel_history[:, 0, :naxis]
        self.ankle_vel_history[:, 0, :naxis] = self.rigid_body_state[
            :, self.rigid_body_idx["right_ankle_roll_link"], 7:10
        ]
        self.ankle_vel_history[:, 1, naxis:] = self.ankle_vel_history[:, 1, :naxis]
        self.ankle_vel_history[:, 1, :naxis] = self.rigid_body_state[
            :, self.rigid_body_idx["left_ankle_roll_link"], 7:10
        ]

        self.step_pos[:, :, :2] = self.rigid_body_state[:, self.feet_ids, :2]
        self.step_pos[:, :, 2] = self.rigid_body_state[:, self.feet_ids, 5]
        # 左上
        self.quadrilateral_point[:, 0, 0] = self.step_pos[:, 0, 0] + 0.17
        self.quadrilateral_point[:, 0, 1] = self.step_pos[:, 0, 1] + 0.04
        # 左下
        self.quadrilateral_point[:, 1, 0] = self.step_pos[:, 0, 0] - 0.08
        self.quadrilateral_point[:, 1, 1] = self.step_pos[:, 0, 1] + 0.04
        # 右下
        self.quadrilateral_point[:, 2, 0] = self.step_pos[:, 1, 0] - 0.08
        self.quadrilateral_point[:, 2, 1] = self.step_pos[:, 1, 1] - 0.04
        # 右上
        self.quadrilateral_point[:, 3, 0] = self.step_pos[:, 1, 0] + 0.17
        self.quadrilateral_point[:, 3, 1] = self.step_pos[:, 1, 1] - 0.04

    def _calculate_foot_states(self, foot_states):
        foot_height_vec = (
            torch.tensor([0.0, 0.0, -0.04]).repeat(self.num_envs, 1).to(self.device)
        )
        rfoot_height_vec_in_world = quat_apply(foot_states[:, 0, 3:7], foot_height_vec)
        lfoot_height_vec_in_world = quat_apply(foot_states[:, 1, 3:7], foot_height_vec)
        foot_states[:, 0, :3] += rfoot_height_vec_in_world
        foot_states[:, 1, :3] += lfoot_height_vec_in_world

        return foot_states

    def _calculate_CoM(self):
        """Calculates the Center of Mass of the robot"""
        self.CoM = (
            self.rigid_body_state[:, :, :3] * self.rigid_body_mass.unsqueeze(1)
        ).sum(dim=1) / self.mass_total

    def _update_command_curriculum(self, env_ids):
        """Implements a curriculum of increasing commands
        Args: env_ids (List[int]): ids of environments being reset
        """
        distance = torch.norm(
            self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1
        )
        move_up = distance > self.terrain.env_length * 0.55
        move_down = (
            distance
            < (
                torch.norm(self.commands[env_ids, :2], dim=1)
                * self.max_episode_length_s
                * 0.45
            )
        ) * ~move_up
        self.termination_levels[env_ids] += 1 * move_up - 1 * move_down
        self.termination_levels[env_ids] = torch.clamp(
            self.termination_levels[env_ids], 0, self.cfg.commands.term_ranges.max_level
        )
        self.term_base_lin_vel_limit[env_ids] = (
            self.cfg.commands.term_ranges.term_base_lin_vel[0]
            + self.termination_levels[env_ids] * self.term_base_lin_vel_step
        )
        self.term_base_ang_vel_limit[env_ids] = (
            self.cfg.commands.term_ranges.term_base_ang_vel[0]
            + self.termination_levels[env_ids] * self.term_base_ang_vel_step
        )
        self.term_projected_gravity_x_limit[env_ids] = (
            self.cfg.commands.term_ranges.term_projected_gravity_x[0]
            + self.termination_levels[env_ids] * self.term_projected_gravity_x_step
        )
        self.term_projected_gravity_y_limit[env_ids] = (
            self.cfg.commands.term_ranges.term_projected_gravity_y[0]
            + self.termination_levels[env_ids] * self.term_projected_gravity_y_step
        )
        self.term_base_pos_limit[env_ids] = (
            self.cfg.commands.term_ranges.term_base_pos[0]
            + self.termination_levels[env_ids] * self.term_base_pos_step
        )
        pass

    def _update_reward_curriculum(self, env_ids):
        """Implements a curriculum of rewards
        Args: env_ids (List[int]): ids of environments being reset
        """
        pass

    def _set_obs_variables(self):
        self.foot_states_right[:, :3] = quat_rotate_inverse(
            self.base_quat, self.foot_states[:, 0, :3] - self.base_pos
        )
        self.foot_states_left[:, :3] = quat_rotate_inverse(
            self.base_quat, self.foot_states[:, 1, :3] - self.base_pos
        )
        self.foot_states_right[:, 3] = wrap_to_pi(
            self.foot_heading[:, 0] - self.base_heading.squeeze(1)
        )
        self.foot_states_left[:, 3] = wrap_to_pi(
            self.foot_heading[:, 1] - self.base_heading.squeeze(1)
        )
        self.phase_sin = torch.sin(self.phase * (1 - self.standing_command_mask))
        self.phase_cos = torch.cos(self.phase * (1 - self.standing_command_mask))
        # print("phase_sin:",self.phase_sin[0,:])
        # print("phase_cos:",self.phase_cos[0,:])
        self.base_lin_vel_world = self.root_states[:, 7:10].clone()

    def check_termination(self):
        """Check if environments need to be reset"""
        # * Termination for contact
        self.term_contact = torch.any(
            (
                torch.norm(
                    self.contact_forces[:, self.termination_contact_indices, :], dim=-1
                )
                > 1.0
            ),
            dim=1,
        )
        self.terminated = self.term_contact

        # * Termination for velocities, orientation, and low height
        self.term_base_lin_vel = torch.any(
            torch.norm(self.base_lin_vel, dim=-1, keepdim=True) > self.term_base_lin_vel_limit, dim=1
        )
        self.terminated |= self.term_base_lin_vel

        self.term_base_ang_vel = torch.any(
            torch.norm(self.base_ang_vel, dim=-1, keepdim=True) > self.term_base_ang_vel_limit, dim=1
        )
        self.terminated |= self.term_base_ang_vel

        self.term_projected_gravity_x = torch.any(
            torch.abs(self.projected_gravity[:, 0:1]) > self.term_projected_gravity_x_limit, dim=1
        )
        self.terminated |= self.term_projected_gravity_x

        self.term_projected_gravity_y = torch.any(
            torch.abs(self.projected_gravity[:, 1:2]) > self.term_projected_gravity_y_limit, dim=1
        )
        self.terminated |= self.term_projected_gravity_y

        self.term_base_pos = torch.any(self.base_pos[:, 2:3] < self.term_base_pos_limit, dim=1)
        self.terminated |= self.term_base_pos

        # * No terminal reward for time-outs
        self.timed_out = self.episode_length_buf > self.max_episode_length

        self.reset_buf = self.terminated | self.timed_out

    def post_physics_step(self):
        super().post_physics_step()
        # self._log_info()

    # * ########################## REWARDS ######################## * #

    # * Regularization rewards * #
    def _reward_actuation_rate(self):
        # Penalize changes in actuations
        nact = self.num_actuators
        dt2 = (self.dt * self.cfg.control.decimation) ** 2
        error = (
            torch.square(
                self.actuation_history[:, :nact]
                - self.actuation_history[:, nact : 2 * nact]
            )
            / dt2
        )
        return -torch.sum(error, dim=1)

    def _reward_actuation_rate2(self):
        # Penalize changes in actuations
        nact = self.num_actuators
        dt2 = (self.dt * self.cfg.control.decimation) ** 2
        error = (
            torch.square(
                self.actuation_history[:, :nact]
                - 2 * self.actuation_history[:, nact : 2 * nact]
                + self.actuation_history[:, 2 * nact :]
            )
            / dt2
        )
        return -torch.sum(error, dim=1)

    def _reward_torques(self):
        # Penalize torques
        return -torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return -torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return -torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity (x:roll, y:pitch)
        return -torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(
            max=0.0
        )  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return -torch.sum(out_of_limits, dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return -torch.sum(
            (
                torch.abs(self.torques)
                - self.torque_limits * self.cfg.rewards.soft_torque_limit
            ).clip(min=0.0),
            dim=1,
        )

    def _reward_joint_regularization(self):
        # Reward joint poses and symmetry
        error = 0.0

        # Yaw joints regularization around 0
        error += self._negsqrd_exp((self.dof_pos[:, 0]) / self.scales["dof_pos"])
        error += self._negsqrd_exp((self.dof_pos[:, 5]) / self.scales["dof_pos"])

        error += self._negsqrd_exp((self.dof_pos[:, 1]) / self.scales["dof_pos"])
        error += self._negsqrd_exp((self.dof_pos[:, 6]) / self.scales["dof_pos"])

        return error / 4 * (1 - self.standing_command_mask.squeeze(1))

    def _reward_stand_joint_regularization(self):
        # Reward joint poses and symmetry
        error = self._negsqrd_exp(self.dof_pos).sum(dim=1)
        return error / 12 * self.standing_command_mask.squeeze(1)

    # * Floating base rewards * #

    def _reward_base_height(self):
        """Reward tracking specified base height"""
        error = (self.cfg.rewards.base_height_target - self.base_height).flatten()
        return self._negsqrd_exp(error)

    def _reward_tracking_lin_vel(self):
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        error *= 1.0 / (1.0 + torch.abs(self.commands[:, :2]))
        return self._negsqrd_exp(error, a=1.0).sum(dim=1) / 2

    def _reward_base_yaw_vel(self):
        error = self.commands[:, 2:3] - self.base_ang_vel[:, 2:3]
        error *= 1.0 / (1.0 + torch.abs(self.commands[:, 2:3]))
        return self._negsqrd_exp(error, a=0.4).sum(dim=1)

    def _reward_base_roll(self):
        # self.base_quat
        threshold = 0.07
        clipped = torch.clamp(torch.abs(self.base_euler[:, 0]) - threshold, 0)
        # mask = torch.max(clipped,0)
        mask = torch.gt(clipped, 0).int()
        error = clipped + mask * threshold
        return self._negsqrd_exp(error, a=0.2)

    def _reward_base_pitch(self):
        # self.base_quat
        threshold = 0.08
        clipped = torch.clamp(torch.abs(self.base_euler[:, 1]) - threshold, 0)
        # mask = torch.max(clipped,0)
        mask = torch.gt(clipped, 0).int()
        error = clipped + mask * threshold
        return self._negsqrd_exp(error, a=0.3)

    # * Stepping Rewards for walking* #
    def _reward_feet_height(self):
        feet_height = self.rigid_body_state[:, self.feet_ids - 1, 2]
        contact = self.contact_forces[:, self.feet_ids, 2] > 0
        in_air = ~contact  # 腾空状态
        height = 0.04
        height_reward = (
            torch.exp(-torch.square((feet_height - (0.041 + 0.005 + 0.02 + height))/0.0044))
        ) * in_air.float()

        return height_reward.sum(dim=1)

    def _reward_feet_contact(self):
        _rew = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        contact = self.contact_forces[:, self.feet_ids, 2] > 0
        # self.contact_queue存放了最近200次(0.2s)的contact
        self.contact_queue = torch.roll(self.contact_queue, 1, dims=0)  # 右滚
        self.contact_queue[:, 0, :] = contact
        sum_contact = torch.sum(self.contact_queue, dim=2)

        # has_true_in_row_zero = torch.any(sum_contact == 0, dim=1)
        has_true_in_row = torch.any(sum_contact == 1, dim=1)
        has_true_in_row_double = torch.any(sum_contact == 2, dim=1)

        pitch_mask = (torch.abs(self.projected_gravity[:, 0])) > (0.5)
        roll_mask = (torch.abs(self.projected_gravity[:, 1])) > (0.5)
        y_vel_mask = (torch.abs(self.base_lin_vel[:, 1])) > (0.1)
        x_vel_mask = (torch.abs(self.base_lin_vel[:, 0])) > (0.1)

        # if not standing command
        mask_walk_single_contact = (self.standing_command_mask.squeeze(1) != 1) & (
            has_true_in_row == 1
        )
        contact_rewards = (
            self.foot_contact[:, 0].int() - self.foot_contact[:, 1].int()
        ) * self.contact_schedule.squeeze(1)
        _rew[mask_walk_single_contact] = contact_rewards[mask_walk_single_contact]
        # if standing command
        mask_stand_double_contact = (self.standing_command_mask.squeeze(1) == 1) & (
            has_true_in_row_double == 1
        )
        mask_stand_single_contact = (self.standing_command_mask.squeeze(1) == 1) & (
            has_true_in_row == 1
        )
        _rew[mask_stand_single_contact] = (
            # 0.1  # * self.dist_norm[mask_stand_single_contact]
            +0.3 * y_vel_mask[mask_stand_single_contact]
            + 0.3 * x_vel_mask[mask_stand_single_contact]
            + 0.3 * pitch_mask[mask_stand_single_contact]
            + 0.3 * roll_mask[mask_stand_single_contact]
        )
        _rew[mask_stand_double_contact] = 1
        # _rew[mask_stand_zero_contact] = -1

        return _rew

    def _reward_feet_airtime(self, play=False):
        """
        通过在每次脚着地时施加-0.4的惩罚来规范踏步频率，
        这可以通过一个积极的奖励成分来抵消，即足部腾空后的秒数(腾空时间)。
        如果没有这个组件，学习到的控制器倾向于采用步进频率在风格上太大的步态，
        这可能是由于这些频率对应于可能的局部最小值。这个分量在站立时是恒定的。
        """
        _rew = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        # if standing command
        if torch.any(self.standing_command_mask.squeeze(1) == 1):
            # print(self.standing_command_mask.squeeze(1) == 1)
            _rew[self.standing_command_mask.squeeze(1) == 1] = 0
        # else
        contact = self.contact_forces[:, self.feet_ids, 2] > 0
        self.contact_filt = torch.logical_and(
            contact, (self.feet_air_time > 0)
        )  # 是否第一次接触，如果接触就为1，不接触就为0
        # print(self.contact_filt[0,:],(self.feet_air_time * self.contact_filt)[0,:])
        self.last_contacts = contact  # 更新上一帧的接触情况
        self.feet_air_time += self.dt
        a = self.dt * self.cfg.commands.ranges.sample_period[0]  # 0.02*17 = 0.34
        b = 0.02
        c = 0.6
        air_time_reward: torch.Tensor = 1 - 1 / (
            1
            + torch.exp(
                -torch.pow(torch.abs(self.feet_air_time * self.contact_filt - a) / b, c)
            )
        )
        self.feet_air_time *= ~contact  # 不接触的话就持续计数，接触就清零
        if torch.any(self.standing_command_mask.squeeze(1) == 0):
            _rew[self.standing_command_mask.squeeze(1) == 0] = air_time_reward[
                self.standing_command_mask.squeeze(1) == 0
            ].sum(dim=1)
        # print(_rew)
        if play:
            return (
                contact,
                self.contact_filt,
                self.feet_air_time,
                air_time_reward,
                _rew,
                self.standing_command_mask.squeeze(1),
            )
        else:
            return _rew

    def _reward_boundary(self):
        mask = ~self.is_point_in_quadrilateral(
            self.quadrilateral_point, self.root_states[:, :2]
        )
        # print(mask[0])
        return mask

    def _reward_hip_pos(self):
        return -torch.sum(torch.square(self.dof_pos[:, [0, 2, 6, 8]]), dim=1)

    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = self.contact_forces[:, self.feet_ids, 2] > 5.0
        foot_speed_norm = torch.norm(
            self.rigid_body_state[:, self.feet_ids, 7:9], dim=2
        )
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)

    def is_point_in_quadrilateral(
        self, quadrilateral_point: torch.Tensor, root_states: torch.Tensor
    ) -> torch.Tensor:
        """
        判断每个环境中的root_states是否在对应的四边形内部。

        参数：
            quadrilateral_point: Tensor，形状 [num_envs, 4, 2]，
                                存放每个环境中四边形的四个顶点，
                                点的顺序依次为左上、左下、右下、右上。
            root_states: Tensor，形状 [num_envs, 2]，每个环境中待检测的点。

        返回：
            inside: Bool型Tensor，形状 [num_envs]，每个元素表示对应环境中
                    root_states 点是否在四边形内部（True表示在内部，False表示不在内部）。
        """
        # 将四边形各个顶点沿第1维滚动（即对每个环境，将每个点的“下一个”顶点计算出来）
        quadr_next = torch.roll(
            quadrilateral_point, shifts=-1, dims=1
        )  # 形状 [num_envs, 4, 2]

        # 计算每条边的向量：edge = next_vertex - current_vertex
        edge_vectors = quadr_next - quadrilateral_point  # 形状 [num_envs, 4, 2]

        # 计算从每个顶点到待判断点的向量：相对向量
        # 注意：root_states的形状 [num_envs,2]扩展成 [num_envs,1,2]，与四边形顶点相减
        rel_vectors = (
            root_states.unsqueeze(1) - quadrilateral_point
        )  # 形状 [num_envs, 4, 2]

        # 计算二维叉积：对于二维向量 (a, b) 和 (c, d)，叉积的标量为 a*d - b*c
        cross_products = (
            edge_vectors[..., 0] * rel_vectors[..., 1]
            - edge_vectors[..., 1] * rel_vectors[..., 0]
        )  # 形状 [num_envs, 4]

        # 判断：如果在某个环境中，所有边的叉积均大于等于0或均小于等于0，则点在内部
        inside = torch.all(cross_products >= 0, dim=1) | torch.all(
            cross_products <= 0, dim=1
        )
        return inside

    # * Stepping Rewards for standing* #

    def _reward_alive(self):
        # Reward for staying alive
        return 1.0

    def _reward_term_contact(self):
        return -self.term_contact.float()

    def _reward_term_base_lin_vel(self):
        return -self.term_base_lin_vel.float()

    def _reward_term_base_ang_vel(self):
        return -self.term_base_ang_vel.float()

    def _reward_term_projected_gravity_x(self):
        return -self.term_projected_gravity_x.float()

    def _reward_term_projected_gravity_y(self):
        return -self.term_projected_gravity_y.float()

    def _reward_term_base_pos(self):
        return -self.term_base_pos.float()

    # ##################### HELPER FUNCTIONS ################################## #

    def smooth_sqr_wave(self, p):
        eps = 0.2
        return torch.sin(p) / torch.sqrt(torch.sin(p) ** 2.0 + eps**2.0)

    def get_euler_xyz_tensor(self, quat):
        r, p, w = get_euler_xyz(quat)
        # stack r, p, w in dim1
        euler_xyz = torch.stack((r, p, w), dim=1)
        euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
        return euler_xyz

    def _log_info(self):
        """Log any information for debugging"""
        self.extras["dof_vel"] = self.dof_vel
        self.extras["step_commands"] = self.step_commands
        self.extras["update_count"] = self.update_count

    def _visualization(self):
        # self.gym.clear_lines(self.viewer)
        # self._draw_heightmap_vis()
        # self._draw_debug_vis()
        # self._draw_velocity_arrow_vis()
        # self._draw_world_velocity_arrow_vis()
        # self._draw_base_pos_vis()
        # self._draw_CoM_vis()
        # self._draw_raibert_vis()
        # self._draw_step_vis()
        # self._draw_step_command_vis()
        ...

    def _draw_debug_vis(self):
        """Draws anything for debugging for humanoid"""
        sphere_origin = gymutil.WireframeSphereGeometry(
            0.02, 4, 4, None, color=(1, 1, 1)
        )
        origins = self.base_pos + quat_apply(
            self.base_quat,
            torch.tensor([0.0, 0.0, 0.5]).repeat(self.num_envs, 1).to(self.device),
        )

        for i in range(self.num_envs):
            env_origin = gymapi.Transform(gymapi.Vec3(*self.env_origins[i]), r=None)
            gymutil.draw_lines(
                sphere_origin, self.gym, self.viewer, self.envs[i], env_origin
            )

    def _draw_velocity_arrow_vis(self):
        """Draws linear / angular velocity arrow for humanoid
        Angular velocity is described by axis-angle representation"""
        origins = self.base_pos + quat_apply(
            self.base_quat,
            torch.tensor([0.0, 0.0, 0.5]).repeat(self.num_envs, 1).to(self.device),
        )
        lin_vel_command = quat_apply(
            self.base_quat,
            torch.cat(
                (
                    self.commands[:, :2],
                    torch.zeros((self.num_envs, 1), device=self.device),
                ),
                dim=1,
            )
            / 5,
        )
        ang_vel_command = quat_apply(
            self.base_quat,
            torch.cat(
                (
                    torch.zeros((self.num_envs, 2), device=self.device),
                    self.commands[:, 2:3],
                ),
                dim=1,
            )
            / 5,
        )
        for i in range(self.num_envs):
            lin_vel_arrow = VelCommandGeometry(
                origins[i], lin_vel_command[i], color=(0, 1, 0)
            )
            ang_vel_arrow = VelCommandGeometry(
                origins[i], ang_vel_command[i], color=(0, 1, 0)
            )
            gymutil.draw_lines(
                lin_vel_arrow, self.gym, self.viewer, self.envs[i], pose=None
            )
            gymutil.draw_lines(
                ang_vel_arrow, self.gym, self.viewer, self.envs[i], pose=None
            )

    def _draw_world_velocity_arrow_vis(self):
        """Draws linear / angular velocity arrow for humanoid
        Angular velocity is described by axis-angle representation"""
        origins = self.base_pos + quat_apply(
            self.base_quat,
            torch.tensor([0.0, 0.0, 0.5]).repeat(self.num_envs, 1).to(self.device),
        )
        lin_vel_command = (
            torch.cat(
                (
                    self.commands[:, :2],
                    torch.zeros((self.num_envs, 1), device=self.device),
                ),
                dim=1,
            )
            / 5
        )
        # ang_vel_command = quat_apply(self.base_quat, torch.cat((torch.zeros((self.num_envs,2), device=self.device), self.commands[:, 2:3]), dim=1)/5)
        for i in range(self.num_envs):
            lin_vel_arrow = VelCommandGeometry(
                origins[i], lin_vel_command[i], color=(0, 1, 0)
            )
            # ang_vel_arrow = VelCommandGeometry(origins[i], ang_vel_command[i], color=(0,1,0))
            gymutil.draw_lines(
                lin_vel_arrow, self.gym, self.viewer, self.envs[i], pose=None
            )
            # gymutil.draw_lines(ang_vel_arrow, self.gym, self.viewer, self.envs[i], pose=None)

    def _draw_base_pos_vis(self):
        sphere_base = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 1))
        sphere_left_hip = gymutil.WireframeSphereGeometry(
            0.02, 4, 4, None, color=(0, 0, 1)
        )
        sphere_right_hip = gymutil.WireframeSphereGeometry(
            0.02, 4, 4, None, color=(1, 0, 0)
        )

        base_projection = self.base_pos[:, :3]
        right_hip_projection = self.right_hip_pos[:, :3]
        left_hip_projection = self.left_hip_pos[:, :3]
        for i in range(self.num_envs):
            base_loc = gymapi.Transform(gymapi.Vec3(*base_projection[i]), r=None)
            gymutil.draw_lines(
                sphere_base, self.gym, self.viewer, self.envs[i], base_loc
            )
            right_hip_loc = gymapi.Transform(
                gymapi.Vec3(*right_hip_projection[i]), r=None
            )
            gymutil.draw_lines(
                sphere_right_hip, self.gym, self.viewer, self.envs[i], right_hip_loc
            )
            left_hip_loc = gymapi.Transform(
                gymapi.Vec3(*left_hip_projection[i]), r=None
            )
            gymutil.draw_lines(
                sphere_left_hip, self.gym, self.viewer, self.envs[i], left_hip_loc
            )

    def _draw_CoM_vis(self):
        sphere_CoM = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 1))
        CoM_projection = self.CoM[:, :3]
        for i in range(self.num_envs):
            CoM_loc = gymapi.Transform(gymapi.Vec3(*CoM_projection[i]), r=None)
            gymutil.draw_lines(sphere_CoM, self.gym, self.viewer, self.envs[i], CoM_loc)

    def _draw_raibert_vis(self):
        sphere_right_raibert = gymutil.WireframeSphereGeometry(
            0.02, 4, 4, None, color=(1, 0, 0)
        )
        sphere_left_raibert = gymutil.WireframeSphereGeometry(
            0.02, 4, 4, None, color=(0, 0, 1)
        )

        for i in range(self.num_envs):
            right_raibert_loc = gymapi.Transform(
                gymapi.Vec3(*self.raibert_heuristic[i, 0]), r=None
            )
            gymutil.draw_lines(
                sphere_right_raibert,
                self.gym,
                self.viewer,
                self.envs[i],
                right_raibert_loc,
            )

            left_raibert_loc = gymapi.Transform(
                gymapi.Vec3(*self.raibert_heuristic[i, 1]), r=None
            )
            gymutil.draw_lines(
                sphere_left_raibert,
                self.gym,
                self.viewer,
                self.envs[i],
                left_raibert_loc,
            )

    def _draw_step_vis(self):
        """Draws current foot steps for humanoid"""
        for i in range(self.num_envs):
            right_step_command = FootStepGeometry(
                self.step_pos[i, 0, :2],
                self.step_pos[i, 0, 2],
                color=(1, 0, 0),
            )  # Right foot: Red
            left_step_command = FootStepGeometry(
                self.step_pos[i, 1, :2],
                self.step_pos[i, 1, 2],
                color=(0, 0, 1),
            )  # Left foot: Blue
            # gymutil.draw_lines(
            #     left_step_command, self.gym, self.viewer, self.envs[i], pose=None
            # )
            # gymutil.draw_lines(
            #     right_step_command, self.gym, self.viewer, self.envs[i], pose=None
            # )

            verts = np.empty((1, 2), dtype=gymapi.Vec3.dtype)
            verts[0][0] = (
                self.step_pos[i, 0, 0] + 0.170,
                self.step_pos[i, 0, 1] + 0.04,
                1e-4,
            )
            verts[0][1] = (
                self.step_pos[i, 1, 0] + 0.170,
                self.step_pos[i, 1, 1] - 0.04,
                1e-4,
            )

            colors = np.empty(1, dtype=gymapi.Vec3.dtype)
            colors[0] = (0, 1, 0)
            self.gym.add_lines(self.viewer, self.envs[i], 1, verts, colors)

            verts = np.empty((1, 2), dtype=gymapi.Vec3.dtype)
            verts[0][0] = (
                self.step_pos[i, 0, 0] - 0.08,
                self.step_pos[i, 0, 1] + 0.04,
                1e-4,
            )
            verts[0][1] = (
                self.step_pos[i, 1, 0] - 0.08,
                self.step_pos[i, 1, 1] - 0.04,
                1e-4,
            )
            colors = np.empty(1, dtype=gymapi.Vec3.dtype)
            colors[0] = (0, 1, 0)
            self.gym.add_lines(self.viewer, self.envs[i], 1, verts, colors)

    def _draw_step_command_vis(self):
        """Draws step command for humanoid"""
        for i in range(self.num_envs):
            right_step_command = FootStepGeometry(
                self.step_commands[i, 0, :2],
                self.step_commands[i, 0, 2],
                color=(1, 0, 0),
            )  # Right foot: Red
            left_step_command = FootStepGeometry(
                self.step_commands[i, 1, :2],
                self.step_commands[i, 1, 2],
                color=(0, 0, 1),
            )  # Left foot: Blue
            gymutil.draw_lines(
                left_step_command, self.gym, self.viewer, self.envs[i], pose=None
            )
            gymutil.draw_lines(
                right_step_command, self.gym, self.viewer, self.envs[i], pose=None
            )


""" Code Explanation
0.
[Axis] X-axis: Red, Y-axis: Green, Z-axis: Blue

1.
self.base_pos = self.root_states[:, 0:3] : position of the base
self.base_quat = self.root_states[:, 3:7] : quaternion of the base
self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10]) : base linear velocity wrt base frame
self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13]) : base angular velocity wrt base frame

2.                                                    
quat_rotate_inverse() : World frame -> Base frame
quat_rotate(), quat_apply() : Base frame -> World frame

3.
self.rigid_body_state : [num_envs, num_bodies, 13] = [num_envs, 21, 13] 
[position | orientation (Quat) | linear velocity | angular velocity]

self._rigid_body_pos = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3]
self._rigid_body_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 7:10]

4.
21 bodies: base / right_hip_yaw / right_hip_abad / right_upper_leg / right_lower_leg / right_foot / left_hip_yaw / left_hip_abad / left_upper_leg / left_lower_leg / left_foot
                / right_shoulder / right_shoulder_2 / right_upper_arm / right_lower_arm / right_hand / left_shoulder / left_shoulder_2 / left_upper_arm / left_lower_arm / left_hand

right_foot[5] / left_foot[10] are end-effector

5.
self.dof_pos : joint position [num_envs, 10]       
self.dof_vel : joint velocity [num_envs, 10]                     
10 dof: 01_right_hip_yaw / 02_right_hip_abad / 03_right_hip_pitch / 04_right_knee / 05_right_ankle
        06_left_hip_yaw / 07_left_hip_abad / 08_left_hip_pitch / 09_left_knee / 10_left_ankle

6.
self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec) : gravity wrt base frame

7.
self.contact_forces : contact forces of each body parts [num_envs, num_bodies, 3] = [num_envs, 21, 3]
Contact forces are only measured when the collision body is defined. 

self.foot_contact : which foot (right & left foot) are in contact with ground [num_envs, 2]

8.
self.feet_ids: right_foot[5], left_foot[10]
self.end_eff_ids: right_foot[5], left_foot[10]

9.
Maximum reward we can get is "Max value of reward function * reward weight".
Since how it records the reward is "value * weight * dt  * (max_episode_length_s / dt) / max_episode_length_s = value * weight"
"""

""" TODO: 
1) Fix foot_reference_trajectory reward. It forces not to do sprint. 
Because the trajectory always start from the previous step command. Gradually increase the rewards.
2) Systematic training curriculum is necessary
"""
