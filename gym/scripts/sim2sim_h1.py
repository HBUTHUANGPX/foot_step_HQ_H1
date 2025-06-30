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
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


from gym.envs import LEGGED_GYM_ROOT_DIR, H1ControllerCfg
import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
import torch
import time
import csv
import pandas as pd

import os
from gym.scripts import pin_mj


class cmd:
    vx = 2.0
    vy = 0.0
    dyaw = 0.0


def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])


def get_obs(data):
    """Extracts an observation from the mujoco data structure"""
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor("angular-velocity").data.astype(np.double)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    state_tau = data.qfrc_actuator.astype(np.double) - data.qfrc_bias.astype(np.double)
    return (q, dq, quat, v, omega, gvec, state_tau)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    # print("p:", (target_q - q) * kp )
    # print("d", (target_dq - dq) * kd)
    return (target_q - q) * kp + (target_dq - dq) * kd


import threading
import queue
import matplotlib.pyplot as plt
import time

data_queue = queue.Queue()
plot_num = 12

# def plot_data(data_queue):
#     print("plot_data")
#     plt.ion()  # 开启交互模式
#     fig, axs = plt.subplots(plot_num, 1, figsize=(10, 12))  # 创建 8 个子图
#     lines = [ax.plot([], [])[0] for ax in axs]  # 初始化每个子图的线条
#     xdata = [[] for _ in range(plot_num)]  # 存储每个子图的 x 数据
#     ydata = [[] for _ in range(plot_num)]  # 存储每个子图的 y 数据


#     while True:
#         if not data_queue.empty():
#             merged_tensor = data_queue.get()
#             # print("bb")
#             for i in range(plot_num):
#                 xdata[i].append(len(xdata[i]))
#                 ydata[i].append(merged_tensor[i].item())
#                 lines[i].set_data(xdata[i], ydata[i])
#                 axs[i].relim()
#                 axs[i].autoscale_view()
#             fig.canvas.draw()
#             fig.canvas.flush_events()
#         else:
#             # print("cc")
#             time.sleep(0.1)
def plot_data(data_queue):
    print("plot_data")
    plt.ion()  # 开启交互模式

    first_flag = 1
    while True:
        if not data_queue.empty():
            merged_tensor = data_queue.get()
            plot_num = merged_tensor.shape[0]
            if first_flag:
                first_flag = 0
                # 计算行数和列数
                rows = math.floor(math.sqrt(plot_num))
                cols = math.ceil(plot_num / rows)

                fig, axs = plt.subplots(rows, cols, figsize=(10, 12))  # 创建子图
                axs = axs.flatten()  # 将二维数组展平成一维数组，方便索引

                lines = [ax.plot([], [])[0] for ax in axs]  # 初始化每个子图的线条
                xdata = [
                    [0 for _ in range(1000)] for _ in range(plot_num)
                ]  # 存储每个子图的 x 数据
                ydata = [[0] * 1000 for _ in range(plot_num)]  # 存储每个子图的 y 数据
            for i in range(plot_num):
                xdata[i].append(len(xdata[i]))
                ydata[i].append(merged_tensor[i].item())
                lines[i].set_data(xdata[i][-1000:], ydata[i][-1000:])
                axs[i].relim()
                axs[i].autoscale_view()
            # print(len(xdata[i]))
            if len(xdata[i]) % 200 == 0:
                fig.canvas.draw()
                fig.canvas.flush_events()

        # else:
        #     time.sleep(0.1)


def run_mujoco(policy, cfg: H1ControllerCfg):
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)

    nq = model.nq  # 关节位置的自由度
    nv = model.nv  # 关节速度的自由度

    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actuators), dtype=np.double)
    action = np.zeros((cfg.env.num_actuators), dtype=np.double)

    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

    count_lowlevel = 0
    count_simlevel = 0

    count_csv = 0

    phase = 0
    step_period = 15  # 38#
    first_flag = 0
    if cfg.sim_config.plot_flag:
        plot_thread = threading.Thread(target=plot_data, args=(data_queue,))
        plot_thread.daemon = True
        plot_thread.start()
    flip_line = int(0.5 * 1 / cfg.sim_config.dt)
    flip_count = 0
    flip_flag = 1
    # for _ in range(int(21)):
    # for _ in range(int(2001)):
    for _ in range(int(1e166)):
        # for _ in range(int(1+20)):
        # Obtain an observation
        q, dq, quat, v, omega, gvec, state_tau = get_obs(data)
        q = q[-cfg.env.num_actuators :]
        dq = dq[-cfg.env.num_actuators :]
        state_tau = state_tau[-cfg.env.num_actuators :]
        if cfg.sim_config.plot_flag:
            # merged_tensor = dq[-6:]
            merged_tensor = target_q[-6:]
            data_queue.put(merged_tensor)
        if count_simlevel % cfg.sim_config.sim_decimation == 0:
            if count_lowlevel % cfg.sim_config.decimation == 0:
                obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
                _q = quat
                _v = np.array([0.0, 0.0, -1.0])
                projected_gravity = quat_rotate_inverse(_q, _v)
                eu_ang = quaternion_to_euler_array(quat)
                eu_ang[eu_ang > math.pi] -= 2 * math.pi

                full_step_period = step_period * 2
                phase += 1 / full_step_period
                # base_ang_vel
                # obs[0, 0:3] = omega * cfg.scaling.base_ang_vel * 0
                obs[0, 0:3] = omega * cfg.scaling.base_ang_vel
                # print("base_ang_vel:\r\n",obs[0, 0:3])
                # projected_gravity
                # obs[0, 3:6] = projected_gravity * cfg.scaling.projected_gravity * 0
                # obs[0, 5] = -1
                obs[0, 3:6] = projected_gravity * cfg.scaling.projected_gravity
                # print("projected_gravity:\r\n",obs[0, 3:6])
                # commands
                obs[0, 6] = cmd.vx * cfg.scaling.commands
                obs[0, 7] = cmd.vy * cfg.scaling.commands
                obs[0, 8] = cmd.dyaw * cfg.scaling.commands
                # print("commands:\r\n",obs[0, 6:9])
                # standing_command_mask
                obs[0, 43] = 0
                # phase_sin
                obs[0, 9] = math.sin(2 * math.pi * phase * (1 - obs[0, 43]))
                # phase_cos
                obs[0, 10] = math.cos(2 * math.pi * phase * (1 - obs[0, 43]))

                # print("phase_sin:\r\n",obs[0, 9])
                # print("phase_cos:\r\n",obs[0, 10])
                # dof_pos
                obs[0, 11:23] = q * cfg.scaling.dof_pos
                # print("dof_pos:\r\n",obs[0, 11:23])
                # dof_vel
                obs[0, 23:35] = dq * cfg.scaling.dof_vel
                # print("dof_vel:\r\n",obs[0, 23:35])
                # foot_states_right foot_states_left
                obs[0, 35:43] = cfg.sim_config.pin_f.get_foot_pos(q)
                # print("foot_states:\r\n", obs[0, 35:43])

                # print("standing_command_mask:\r\n",obs[0, 43])

                obs = np.clip(
                    obs,
                    -20,
                    20,
                )
                # obs *= 0
                hist_obs.append(obs)
                hist_obs.popleft()

                policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
                for i in range(cfg.env.frame_stack):
                    policy_input[
                        0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs
                    ] = hist_obs[i][0, :]

                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
                # print("policy_input: ",policy_input)
                # print("action: ",action)
                action = np.clip(
                    action,
                    -cfg.scaling.clip_actions,
                    cfg.scaling.clip_actions,
                )
                target_q = action
                if first_flag == 1:
                    target_q = action  # * np.array(cfg.control.actuation_scale, dtype=np.double)
                else:
                    target_q = 0 * action
                # print("obs: ",obs)
                # print("action: ",action)
            target_dq = np.zeros((cfg.env.num_actuators), dtype=np.double)
            # target_q *= 0
            # target_q[0] = 0.2 * flip_flag
            # target_q[0]*=0
            # target_q[0+6]*=0
            tau = pd_control(
                target_q, q, cfg.robot_config.kps, target_dq, dq, cfg.robot_config.kds
            )  # Calc torques
            # t_max = (np.abs(dq) - cfg.motor.b) / cfg.motor.k
            # print("q: ",q)
            # print("tau: ",tau)
            tau = np.clip(
                tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit
            )  # Clamp torques
            # for i in range(6):
            #     tmptau = tau[i]
            #     tau[i] = tau[i + 6]
            #     tau[i + 6] = tmptau
            count_lowlevel += 1

        # print(tau)
        count_simlevel += 1
        flip_count += 1
        if flip_count == flip_line:
            flip_count *= 0
            flip_flag *= -1

        data.ctrl = tau
        mujoco.mj_step(model, data)
        viewer.render()
        first_flag = 1
    viewer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deployment script.")
    parser.add_argument(
        "--load_model",
        type=str,
        required=False,
        help="Run to load from.",
        default=f"{LEGGED_GYM_ROOT_DIR}/logs/U_H1_R/exported/policy_lstm_1.pt",
    )
    parser.add_argument("--terrain", action="store_true", help="terrain or plane")
    args = parser.parse_args()

    class Sim2simCfg(H1ControllerCfg):
        class env:
            frame_stack = 1
            num_actuators = 12
            num_single_obs = 44
            num_observations = 44

        class sim_config:
            mujoco_model_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/h1_2_description/mjcf/h1_12dof_release_rl.xml"
            urdf_path = (
                f"{LEGGED_GYM_ROOT_DIR}/resources/robots/h1_2_description/h1_2.urdf"
            )
            pin_f = pin_mj(urdf_path)
            sim_duration = 60.0
            dt = 0.002
            decimation = 10
            sim_decimation = 1
            plot_flag = False  # True , False

        class robot_config:
            # mujoco sim2sim config
            kps = np.array([100, 150, 150, 300, 40, 10] * 2, dtype=np.double)
            kds = np.array([25.5, 18, 12, 16, 3.0, 0.4] * 2, dtype=np.double)
            # kds = np.array([8.5, 16, 16, 16, 3.0, 0.4] * 2, dtype=np.double)
            # isaacgym train config
            # kps = np.array([200, 200, 200, 300, 40, 40] * 2, dtype=np.double)
            # kds = np.array([2.5, 2.5, 2.5, 4, 2.0, 2.0] * 2, dtype=np.double)
            print(kps)
            print(kds)
            tau_limit = np.array([200, 200, 200, 300, 60, 40] * 2, dtype=np.double)

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())
