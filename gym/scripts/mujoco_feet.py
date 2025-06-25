import numpy as np
import pinocchio as pin
from pinocchio.utils import zero


class pin_mj:
    foot_fl_pose_world: np.ndarray
    foot_fr_pose_world: np.ndarray

    def __init__(self, _URDF_PATH=""):
        # ========== 1. 准备Pinocchio模型 ==========
        URDF_PATH = _URDF_PATH
        self.model, self.data = self.setup_pinocchio_from_urdf(
            URDF_PATH, with_free_flyer=True
        )
        self.base_pos_world = np.array([0.0, 0.0, 0.0], dtype=float)
        self.base_quat_world = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self.l_ankle_roll_link_id = self.model.getFrameId("left_ankle_roll_link")
        self.r_ankle_roll_link_id = self.model.getFrameId("right_ankle_roll_link")

    def setup_pinocchio_from_urdf(self, urdf_path, mesh_dir=None, with_free_flyer=True):
        """
        :param urdf_path: 你的URDF文件路径
        :param mesh_dir: 资源文件夹(如果URDF中有mesh引用)，可为 None
        :param with_free_flyer: 是否使用浮动基 (True或False)
        """
        if with_free_flyer:
            if mesh_dir is not None:
                # 传 package_dirs=[mesh_dir]，再传 root_joint=JointModelFreeFlyer
                model = pin.buildModelFromUrdf(
                    urdf_path, [mesh_dir], pin.JointModelFreeFlyer()  # 注意放进列表
                )
            else:
                # 不需要 mesh_dir，则直接传 root_joint
                model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        else:
            # 固定基版本
            if mesh_dir is not None:
                model = pin.buildModelFromUrdf(urdf_path, [mesh_dir])
            else:
                model = pin.buildModelFromUrdf(urdf_path)

        data = model.createData()
        return model, data

    def mujoco_to_pinocchio(self, base_pos, base_quat, joint_angles, model, data):
        """
        将从Mujoco获取的机器人状态(基座位置、姿态、关节角)赋值到Pinocchio中。
        base_pos: np.array([x, y, z]) 基座在世界坐标系的位置
        base_quat: np.array([w, x, y, z]) 基座在世界坐标系的四元数 (Pinocchio默认的四元数顺序同为 [w,x,y,z])
        joint_angles: np.array([...]) 机器人关节角，长度为model.nq - 7(若有浮动基), 或 model.nq(若固定基)
        model, data: Pinocchio的model和data
        """
        q = zero(model.nq)  # 广义坐标 [7 + nJoints] (若 free-flyer)

        # 如果是浮动基模式，则前7维为 [x, y, z, q_w, q_x, q_y, q_z]
        # 注意：Pinocchio中free-flyer的顺序约定是 [xyz, qwxyz]
        # 若是固定基，则model.nq == 机器人关节数，无需设置基座
        if model.joints[1].shortname() == "JointModelFreeFlyer":
            q[0:3] = base_pos
            q[3:7] = base_quat  # [w, x, y, z]
            # 后面是机器人关节
            q[7:] = joint_angles
        else:
            # 如果是固定基模型，则整段q都是关节
            q[:] = joint_angles

        # forwardKinematics: 计算各链接在世界坐标系下的位姿
        pin.forwardKinematics(model, data, q)
        # forwardGeometry(或 updateFramePlacements) 通常可以帮助更新 frame 的位姿
        pin.updateFramePlacements(model, data)

        return q

    def get_foot_pos(self, joint_angles):
        q_pin = self.mujoco_to_pinocchio(
            self.base_pos_world,
            self.base_quat_world,
            joint_angles,
            self.model,
            self.data,
        )
        # print("foot_fl_id: ",self.l_ankle_roll_link_id)
        # frame 变换: data.oMf[frame_id]
        self.foot_fl_pose_world: np.ndarray = self.data.oMf[
            self.l_ankle_roll_link_id
        ].translation
        self.foot_fl_rot_world: np.ndarray = self.data.oMf[
            self.l_ankle_roll_link_id
        ].rotation
        l_yaw = np.arctan2(self.foot_fl_rot_world[1, 0], self.foot_fl_rot_world[0, 0])
        self.foot_fr_pose_world: np.ndarray = self.data.oMf[
            self.r_ankle_roll_link_id
        ].translation
        self.foot_fr_rot_world: np.ndarray = self.data.oMf[
            self.r_ankle_roll_link_id
        ].rotation
        r_yaw = np.arctan2(self.foot_fr_rot_world[1, 0], self.foot_fr_rot_world[0, 0])
        # print("foot_fl in base frame => translation: \n  ", type(foot_fl_pose_world))
        # print("foot_fl in base frame => translation: \n  ", self.foot_fl_pose_world)
        # print("foot_fl in base frame => rotation: \n  ", self.foot_fl_rot_world)
        # print("foot_fl in base frame => yaw: \n  ", l_yaw)
        # print("foot_fr in base frame => translation: \n  ", self.foot_fr_pose_world)
        # print("foot_fr in base frame => rotation: \n  ", self.foot_fr_rot_world)
        # print("foot_fr in base frame => yaw: \n  ", r_yaw)
        # foot_concat_1d = np.concatenate((self.foot_fr_pose_world, self.foot_fl_pose_world), axis=0)
        foot_concat_1d = np.concatenate(
            (self.foot_fl_pose_world,[l_yaw] ,self.foot_fr_pose_world,[r_yaw]), axis=0
        )
        # print("foot_concat_1d in base frame => translation: \n  ", foot_concat_1d)

        return foot_concat_1d


if __name__ == "__main__":
    base_path = "/home/hpx/HPX_Loco/foot_step/resources/robots"
    # robot_patch = "/hi_12dof_250108_4/urdf/hi_12dof_250108_4_rl_3.urdf"
    robot_patch = "/h1_2_description/h1_2.urdf"
    a = pin_mj(base_path + robot_patch)
    arr = np.array([0] * (12), dtype=float)
    arr[0] = 0.1
    hh = a.get_foot_pos(arr)
    # conda install pinocchio -c conda-forge
