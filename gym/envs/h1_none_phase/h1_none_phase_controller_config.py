"""
Configuration file for "fixed arm" (FA) humanoid environment
with potential-based rewards implemented
"""

import torch
from gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotRunnerCfg


class H1_NPControllerCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actuators = 12
        episode_length_s = 25  # 100
        action_scale = 1.0

    class terrain(LeggedRobotCfg.terrain):
        curriculum = False # True, False
        mesh_type = "plane"  # 'plane' 'heightfield' 'trimesh'
        horizontal_scale = 0.1
        vertical_scale = 0.005  # [m]
        selected = False  # True, False
        measure_heights = False
        # terrain_kwargs = {"type": "stepping_stones"}
        # terrain_kwargs = {'type': 'random_uniform'}
        # terrain_kwargs = {'type': 'gap'}
        # difficulty = 0.35 # For gap terrain
        # platform_size = 5.5 # For gap terrain
        difficulty = 5.0  # For rough terrain
        terrain_length = 8.0  # For rough terrain
        terrain_width = 8.0  # For rough terrain
        platform_size = 5.0
        # terrain types: [pyramid_sloped, random_uniform, stairs down, stairs up, discrete obstacles, stepping_stones, gap, pit]
        num_rows = 20  # number of terrain rows (levels)
        max_init_terrain_level = num_rows-1 # starting curriculum state
        num_cols = 10  # number of terrain cols (types)
        terrain_proportions = [0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]
        slope_treshold = (
            1  # slopes above this threshold will be corrected to vertical surfaces
        )

    class init_state(LeggedRobotCfg.init_state):
        # reset_mode = 'reset_to_range' # 'reset_to_basic'
        reset_mode = "reset_to_basic"  # 'reset_to_basic'
        pos = [0.0, 0.0, 1.0243]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        # ranges for [x, y, z, roll, pitch, yaw]
        root_pos_range = [
            [0.0, 0.0],  # x
            [0.0, 0.0],  # y
            [1.0243, 1.0243],  # z
            [-torch.pi / 10, torch.pi / 10],  # roll
            [-torch.pi / 10, torch.pi / 10],  # pitch
            [-torch.pi / 10, torch.pi / 10],  # yaw
        ]

        # ranges for [v_x, v_y, v_z, w_x, w_y, w_z]
        root_vel_range = [
            [-0.5, 0.5],  # x
            [-0.5, 0.5],  # y
            [-0.5, 0.5],  # z
            [-0.5, 0.5],  # roll
            [-0.5, 0.5],  # pitch
            [-0.5, 0.5],  # yaw
        ]

        default_joint_angles = {
            "left_hip_yaw_joint": 0.0,
            "left_hip_pitch_joint": 0,
            "left_hip_roll_joint": 0,
            "left_knee_joint": 0,
            "left_ankle_pitch_joint": 0,
            "left_ankle_roll_joint": 0.0,
            "right_hip_yaw_joint": 0,
            "right_hip_pitch_joint": 0,
            "right_hip_roll_joint": 0,
            "right_knee_joint": 0,
            "right_ankle_pitch_joint": 0,
            "right_ankle_roll_joint": 0,
        }

        dof_pos_range = {
            "left_hip_yaw_joint": [-0.1, 0.1],
            "left_hip_pitch_joint": [-0.1, 0.1],
            "left_hip_roll_joint": [-0.1, 0.1],
            "left_knee_joint": [-0.1, 0.1],
            "left_ankle_pitch_joint": [-0.1, 0.1],
            "left_ankle_roll_joint": [-0.1, 0.1],
            "right_hip_yaw_joint": [-0.1, 0.1],
            "right_hip_pitch_joint": [-0.1, 0.1],
            "right_hip_roll_joint": [-0.1, 0.1],
            "right_knee_joint": [-0.1, 0.1],
            "right_ankle_pitch_joint": [-0.1, 0.1],
            "right_ankle_roll_joint": [-0.1, 0.1],
        }

        dof_vel_range = {
            "left_hip_yaw_joint": [-0.1, 0.1],
            "left_hip_pitch_joint": [-0.1, 0.1],
            "left_hip_roll_joint": [-0.1, 0.1],
            "left_knee_joint": [-0.1, 0.1],
            "left_ankle_pitch_joint": [-0.1, 0.1],
            "left_ankle_roll_joint": [-0.1, 0.1],
            "right_hip_yaw_joint": [-0.1, 0.1],
            "right_hip_pitch_joint": [-0.1, 0.1],
            "right_hip_roll_joint": [-0.1, 0.1],
            "right_knee_joint": [-0.1, 0.1],
            "right_ankle_pitch_joint": [-0.1, 0.1],
            "right_ankle_roll_joint": [-0.1, 0.1],
        }

    class control(LeggedRobotCfg.control):
        # stiffness and damping for joints
        stiffness = {
            "left_hip_yaw_joint": 100.0,
            "left_hip_pitch_joint": 150,
            "left_hip_roll_joint": 150,
            "left_knee_joint": 300,
            "left_ankle_pitch_joint": 40,
            "left_ankle_roll_joint": 10.0,
            "right_hip_yaw_joint": 100,
            "right_hip_pitch_joint": 150,
            "right_hip_roll_joint": 150,
            "right_knee_joint": 300,
            "right_ankle_pitch_joint": 40,
            "right_ankle_roll_joint": 10,
        }
        damping = {
            "left_hip_yaw_joint": 10.5,
            "left_hip_pitch_joint": 18,
            "left_hip_roll_joint": 12,
            "left_knee_joint": 16.0,
            "left_ankle_pitch_joint": 3.0,
            "left_ankle_roll_joint": 0.5,
            "right_hip_yaw_joint": 10.5,
            "right_hip_pitch_joint": 18,
            "right_hip_roll_joint": 12,
            "right_knee_joint": 16.0,
            "right_ankle_pitch_joint": 3.0,
            "right_ankle_roll_joint": 0.5,
        }
        exp_avg_decay = None
        decimation = 10

    class sim(LeggedRobotCfg.sim):
        dt = 0.002

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 16
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            num_position_iterations = 4
            num_velocity_iterations = 0

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.0
        num_commands = 3
        resampling_time = 13.0  # 5.

        succeed_step_radius = 0.03
        succeed_step_angle = 10
        apex_height_percentage = 0.15

        sample_angle_offset = 20
        sample_radius_offset = 0.05

        dstep_length = 0.5
        dstep_width = 0.3

        class ranges(LeggedRobotCfg.commands.ranges):
            # TRAINING STEP COMMAND RANGES #
            sample_period = [20, 21]  # [20, 21] # equal to gait frequency
            dstep_width = [0.326, 0.326]  # [0.2, 0.4] # min max [m]

            lin_vel_x = [-2.0, 2.0]  # [-3.0, 3.0] # min max [m/s]
            lin_vel_y = 0.5  # 1.5   # min max [m/s]
            # yaw_vel = 0.0  # min max [rad/s]
            yaw_vel = 3.0  # min max [rad/s]
        class term_ranges:
            max_level = 10
            term_base_lin_vel = [10.0,20.0]
            term_base_ang_vel = [5.0,10.0]
            term_projected_gravity_x = [0.5,1.0]
            term_projected_gravity_y = [0.5,1.0]
            term_base_pos = [0.7,0.3]
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True  # True, False
        friction_range = [0.1, 0.4]

        randomize_base_mass = True  # True, False
        added_mass_range = [-1.0, 1.0]

        push_robots = True# True, False
        push_interval_s = 4.0
        max_push_vel_x = 0.6
        max_push_vel_y = 0.2

        # Add DR for rotor inertia and angular damping

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/h1_2_description/h1_2.urdf"
        keypoints = ["pelvis"]
        end_effectors = ["right_ankle_roll_link", "left_ankle_roll_link"]
        foot_name = "ankle_roll"
        terminate_after_contacts_on = [
            "pelvis",
            "left_hip_yaw_link",
            "left_hip_pitch_link",
            "left_hip_roll_link",
            "left_knee_link",
            "right_hip_yaw_link",
            "right_hip_pitch_link",
            "right_hip_roll_link",
            "right_knee_link",
        ]

        disable_gravity = False
        disable_actuations = False
        disable_motors = False

        # (1: disable, 0: enable...bitwise filter)
        self_collisions = 0
        collapse_fixed_joints = True
        flip_visual_attachments = False

        # Check GymDofDriveModeFlags
        # (0: none, 1: pos tgt, 2: vel target, 3: effort)
        default_dof_drive_mode = 3

        angular_damping = 0.1
        rotor_inertia = [
            0.07920,  # RIGHT LEG
            0.07920,
            0.07920,
            0.07920,
            0.07920,
            0.07920,
            0.07920,  # LEFT LEG
            0.07920,
            0.07920,
            0.07920,
            0.07920,
            0.07920,
        ]
        apply_humanoid_jacobian = True  # True, False

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 1.0243
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.8
        max_contact_force = 1500.0

        curriculum = False
        only_positive_rewards = False
        tracking_sigma = 0.25

        class weights(LeggedRobotCfg.rewards.weights):
            # * Regularization rewards * #
            actuation_rate = 0.01 
            actuation_rate2 = 0.01
            torques = 1e-4
            dof_vel = 1e-3
            lin_vel_z = 1e-1
            ang_vel_xy = 1e-2
            dof_pos_limits = 10
            torque_limits = 1e-2
            joint_regularization = 1.0
            stand_joint_regularization = 1.0

            # * Floating base rewards * #
            base_height = 0.1
            tracking_lin_vel = 4.0
            base_yaw_vel = 12.0  # 6.0
            base_roll = 5.0
            base_pitch = 5.0

            # * Stepping rewards * #
            feet_contact = 7.0
            feet_airtime = 20.0
            feet_height = 6.0# 2.0 10.0 

            boundary = -10
            hip_pos = 10.0
            # alive = 5

        class termination_weights(LeggedRobotCfg.rewards.termination_weights):
            termination = 1.0
            term_contact = 1.0
            term_base_lin_vel = 1.0
            term_base_ang_vel = 1.0
            term_projected_gravity_x = 1.0
            term_projected_gravity_y = 1.0
            term_base_pos = 1.0

    class scaling(LeggedRobotCfg.scaling):
        base_height = 1.0
        base_lin_vel = 1.0  # .5
        base_ang_vel = 1.0  # 2.
        projected_gravity = 1.0
        foot_states_right = 1.0
        foot_states_left = 1.0
        dof_pos = 1.0
        dof_vel = 1.0  # .1
        dof_pos_target = dof_pos  # scale by range of motion

        # Action scales
        commands = 1.0
        clip_actions = 10.0


class H1_NPControllerRunnerCfg(LeggedRobotRunnerCfg):
    do_wandb = True
    seed = 1

    class policy(LeggedRobotRunnerCfg.policy):
        init_noise_std = 1.0
        # origin
        # actor_hidden_dims = [1024,512,256,64]
        # critic_hidden_dims = [1024,512,256,64]
        # test 1
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        # (elu, relu, selu, crelu, lrelu, tanh, sigmoid)
        activation = "tanh"
        normalize_obs = False  # True, False

        rnn_type = "lstm"
        rnn_hidden_size = 64
        rnn_num_layers = 1
        actor_obs = [
            # "base_heading",
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "phase_sin",
            "phase_cos",
            "dof_pos",
            "dof_vel",
            "foot_states_right",
            "foot_states_left",
            "standing_command_mask",
        ]

        critic_obs = actor_obs + [
            "base_height",
            "base_lin_vel",
        ]

        actions = ["dof_pos_target"]

        class noise:
            base_height = 0.05
            base_lin_vel = 0.05
            base_lin_vel_world = 0.05
            base_heading = 0.01
            base_ang_vel = 0.15
            projected_gravity = 0.15
            foot_states_right = 0.01
            foot_states_left = 0.01
            step_commands_right = 0.05
            step_commands_left = 0.05
            commands = 0.1
            dof_pos = 0.05
            dof_vel = 0.5
            foot_contact = 0.1

    class algorithm(LeggedRobotRunnerCfg.algorithm):
        class PPO:
            # algorithm training hyperparameters
            value_loss_coef = 1.0
            use_clipped_value_loss = True
            clip_param = 0.2
            entropy_coef = 0.01
            num_learning_epochs = 5
            num_mini_batches = 4  # minibatch size = num_envs*nsteps/nminibatches
            learning_rate = 1.0e-5
            schedule = "adaptive"  # could be adaptive, fixed
            gamma = 0.99
            lam = 0.95
            desired_kl = 0.01
            max_grad_norm = 1.0

    class runner(LeggedRobotRunnerCfg.runner):
        # policy_class_name = "ActorCritic"
        policy_class_name = "ActorCriticRecurrent"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 15001
        run_name = "HQ"
        experiment_name = "U_H1_NP"
        save_interval = 50
        plot_input_gradients = False
        plot_parameter_gradients = False
