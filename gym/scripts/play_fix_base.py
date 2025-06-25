from gym import LEGGED_GYM_ROOT_DIR
import os
import time
from colorama import Fore

import isaacgym
from isaacgym import gymapi
from gym.envs import *
from gym.utils import (
    get_args,
    export_policy_as_jit,
    task_registry,
    Logger,
    VisualizationRecorder,
    ScreenShotter,
    AnalysisRecorder,
    CSVLogger,
    DictLogger,
    SuccessRater,
)
from gym.scripts.plotting import LivePlotter
from gym.envs import H1Controller
import numpy as np
import torch

import threading
import queue
import matplotlib.pyplot as plt
import time
import cv2
from datetime import datetime
from video_recorder import VideoRecorder

data_queue = queue.Queue()
plot_num = 9


def plot_data(data_queue):
    print("plot_data")
    plt.ion()  # 开启交互模式
    fig, axs = plt.subplots(plot_num, 1, figsize=(10, 12))  # 创建 8 个子图
    lines = [ax.plot([], [])[0] for ax in axs]  # 初始化每个子图的线条
    xdata = [[] for _ in range(plot_num)]  # 存储每个子图的 x 数据
    ydata = [[] for _ in range(plot_num)]  # 存储每个子图的 y 数据

    while True:
        if not data_queue.empty():
            merged_tensor = data_queue.get()
            # print("bb")
            for i in range(plot_num):
                xdata[i].append(len(xdata[i]))
                ydata[i].append(merged_tensor[i].item())
                lines[i].set_data(xdata[i], ydata[i])
                axs[i].relim()
                axs[i].autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
        else:
            # print("cc")
            time.sleep(0.1)


def play(args):
    env: H1Controller
    env_cfg, train_cfg = task_registry.get_cfgs(args)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 9)
    env_cfg.env.env_spacing = 2.0
    env_cfg.env.episode_length_s = int(1e7)  # 5, int(1e7)
    env_cfg.seed = 1
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.terrain.mesh_type = "plane" 
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.num_rows = 1
    # env_cfg.terrain.terrain_proportions = [1.0, 0., 0., 0., 0., 0.0, 0.0]
    env_cfg.domain_rand.push_robots = False  # True
    env_cfg.init_state.reset_mode = (
        "reset_to_basic"  # 'reset_to_basic', 'reset_to_range'
    )
    env_cfg.commands.resampling_time = -1  # -1, 3, 5, 15
    env_cfg.commands.curriculum = False  # True, False
    # env_cfg.viewer.pos = [0, -1.5, 1] # [0, -3.5, 3]
    env_cfg.viewer.pos = [0, -2.5, 0.5]  # [0, -3.5, 3]
    # env_cfg.viewer.lookat = [0, 0, 0] # [1, 1.5, 0]
    env_cfg.viewer.lookat = [0, 0.5, 0]  # [1, 1.5, 0]
    env_cfg.commands.ranges.lin_vel_x = [0.0, 1.0]  # [0., 0.], [-2., 2.]
    env_cfg.commands.ranges.lin_vel_y = 0.0  # 0., 1.
    env_cfg.commands.ranges.yaw_vel = 0.0  # 0., 1.

    env_cfg.commands.adjust_step_command = False  # True, False
    env_cfg.commands.adjust_prob = 0.05
    env_cfg.commands.sample_angle_offset = 20
    env_cfg.commands.sample_radius_offset = 0.05  # 0.05

    env_cfg.commands.ranges.dstep_width = [0.3, 0.3]
    # env_cfg.asset.fix_base_link = True
    # * prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # * load policy
    train_cfg.runner.resume = True
    policy_runner, train_cfg, load_run, ckp = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, play_flag=True
    )
    policy_runner.alg.actor_critic.eval()

    # * export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, "exported"
        )
        export_policy_as_jit(policy_runner.alg.actor_critic, path)
        print("Exported policy as jit script to: ", path)

    start = time.time()
    robot_index = 0  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    default_camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1.0, 1.0, 0.0])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)

    # log_states = [ "root_states", "step_length", "step_width" ] # "base_lin_vel", "base_pos", "dof_pos",
    # log_commands = [ "commands", "dstep_length", "dstep_width"] # "step_commands"
    log_states = ["foot_air_time"]  # "base_lin_vel", "base_pos", "dof_pos",
    log_commands = []  # "step_commands"
    log_rewards = []  # [ "wrong_contact", "step_location", ]
    max_it = int(1e6)
    screenshotter = ScreenShotter(
        env, train_cfg.runner.experiment_name, policy_runner.log_dir
    )

    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[1], camera_properties)
        camera_offset = gymapi.Vec3(1, -1, 0.5)
        camera_rotation = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(-0.3, 0.2, 1), np.deg2rad(135)
        )
        actor_handle = env.gym.get_actor_handle(env.envs[1], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[1], actor_handle, 0)
        env.gym.attach_camera_to_body(
            h1,
            env.envs[1],
            body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION,
        )
        # 初始化视频录制器
        print(load_run)
        print("_{}".format(ckp))
        video_recorder = VideoRecorder(
            path=load_run + "/recordings",
            tag=None,
            video_name="video_{}".format(ckp),
            fps=int(1 / env.dt),
            compress=True,
        )

    print("============start==============")
    plot_thread = threading.Thread(target=plot_data, args=(data_queue,))
    plot_thread.daemon = True
    # plot_thread.start()
    scale = 0.2
    env.commands[:, :] = 0
    for i in range(max_it):
    # for i in range(2):
        actions = policy_runner.get_inference_actions()
        # print("actions: ",actions[0,:])
        # actions*=0
        # actions[:,2]=0.2
        # actions *= 0
        # # print(actions.size(),env.phase.size())
        # p = env.phase[:,0]
        # actions[:, 1] = torch.sin(2.0 * torch.pi * p) * scale
        # actions[:, 3] = -2 * torch.sin(2.0 * torch.pi * p) * scale
        # actions[:, 4] = torch.sin(2.0 * torch.pi * p) * scale
        # actions[:, 1 + 6] = -torch.sin(2.0 * torch.pi * p) * scale
        # actions[:, 3 + 6] = 2 * torch.sin(2.0 * torch.pi * p) * scale
        # actions[:, 4 + 6] = -torch.sin(2.0 * torch.pi * p) * scale
        policy_runner.set_actions(actions)
        env.step()
        policy_runner.reset_envs()
        # print("base_ang_vel:\r\n",env.base_ang_vel[0,:])
        # print("projected_gravity:\r\n",env.projected_gravity[0,:])
        # print("commands:\r\n",env.commands[0,:])
        # print("phase_sin:\r\n",env.phase_sin[0,:])
        # print("phase_cos:\r\n",env.phase_cos[0,:])
        # print("dof_pos:\r\n",env.dof_pos[0,:])
        # print("dof_vel:\r\n",env.dof_vel[0,:])
        # print("foot_states_right:\r\n",env.foot_states_right[0,:])
        # print("foot_states_left:\r\n",env.foot_states_left[0,:])
        # print("standing_command_mask:\r\n",env.standing_command_mask[0,:])
        # if env.screenshot:
        #     image = env.gym.get_camera_image(env.sim, env.envs[0], env.camera_handle, isaacgym.gymapi.IMAGE_COLOR)
        #     image = image.reshape(image.shape[0], -1, 4)[..., :3]
        #     screenshotter.screenshot(image)
        #     env.screenshot = False

        if CUSTOM_COMMANDS:
            # * Scenario 1 (For flat terrain)
            if (i + 1) == 100:
                env.commands[:, 0] = 2.0
                print("vx = ", env.commands[0, 0])
            elif (i + 1) == 600:
                env.commands[:, 0] = -2.0
                print("vx = ", env.commands[0, 0])
            elif (i + 1) == 900:
                env.commands[:, 0] = 0.0
                env.commands[:, 1] = 1.0
                print("vy = ", env.commands[0, 1])
            elif (i + 1) == 1200:
                env.commands[:, 0] = 0.0
                env.commands[:, 1] = -1.0
                print("vy = ", env.commands[0, 1])
            elif (i + 1) == 1500:
                env.commands[:, 0] = 0.0
                env.commands[:, 1] = 0.0
                env.commands[:, 2] = 1.0
                print("wz = ", env.commands[0, 2])
            elif (i + 1) == 1800:
                env.commands[:, 0] = 0.0
                env.commands[:, 1] = 0.0
                env.commands[:, 2] = -1.0
                print("wz = ", env.commands[0, 2])
        # foot_contact, foot_air_time, air_mask, time_rew, rew = env._reward_air_time(
        #     debug=True
        # )
        # merged_tensor = torch.cat(
        #     [foot_contact, foot_air_time, air_mask, time_rew, rew.unsqueeze(1)], dim=1
        # )[0, :]
        # data_queue.put(merged_tensor)

        if RENDER:
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            img = env.gym.get_camera_image(env.sim, env.envs[1], h1, gymapi.IMAGE_COLOR)
            img = np.reshape(img, (1080, 1920, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_rgb = img[..., :3]  # 移除alpha通道，获取RGB图像
            video_recorder(img_rgb)  # 录制当前帧

    if RENDER:
        video_recorder.stop()  # 停止录制并保存视频


if __name__ == "__main__":
    EXPORT_POLICY = True  # True, False
    CUSTOM_COMMANDS = False  # True, False
    MOVE_CAMERA = False  # True, False
    LIVE_PLOT = False  # True, False
    RECORD_FRAMES = True  # True, False
    SAVE_CSV = False  # True, False
    SAVE_DICT = False  # True, False
    CHECK_SUCCESS_RATE = False  # True, False
    args = get_args()
    RENDER = False
    # # * custom loading
    # args.load_files = True # True, False
    # args.load_run = 'Feb06_00-27-24_sf' # load run name
    # args.checkpoint = '1000'

    play(args)
