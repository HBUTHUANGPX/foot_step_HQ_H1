
from isaacgym.torch_utils import *

import torch
# 随机生成数据
batch_size = 10
q = torch.randn(batch_size, 4)
q = q / torch.norm(q, dim=-1, keepdim=True)  # 归一化四元数
v_world = torch.randn(batch_size, 3)

# 正逆旋转
v_body = quat_rotate_inverse(q, v_world)
v_world_recovered = quat_rotate(q, v_body)

# 验证
assert torch.allclose(v_world, v_world_recovered, atol=1e-6), "Rotation failed!"
print("Rotation test passed!")