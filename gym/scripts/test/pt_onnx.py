from gym.envs import LEGGED_GYM_ROOT_DIR
import argparse
import torch
import onnxruntime as ort
import numpy as np


class ModelDeployer:
    def __init__(self, model_path, use_onnx=False, device="cpu"):
        """
        初始化部署器
        :param model_path: 模型路径(.pt或.onnx)
        :param use_onnx: 是否使用ONNX
        :param device: 计算设备(cpu/cuda)
        """
        self.use_onnx = use_onnx
        self.device = device

        if use_onnx:
            # 初始化ONNX运行时
            providers = (
                ["CPUExecutionProvider"]
                if device == "cpu"
                else ["CUDAExecutionProvider"]
            )
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.hidden_in_name = self.session.get_inputs()[1].name
            self.cell_in_name = self.session.get_inputs()[2].name

        else:
            # 初始化PyTorch模型
            self.model = torch.jit.load(model_path)
            self.model.to(device)
            self.model.eval()

    def predict(self, input_data, hidden_in_data, cell_in_data):
        """
        统一推理接口
        :param input_data: 输入张量(支持torch.Tensor或np.array)
        """
        if self.use_onnx:
            # ONNX推理路径
            if isinstance(input_data, torch.Tensor):
                input_data = input_data.cpu().numpy()
            if isinstance(hidden_in_data, torch.Tensor):
                hidden_in_data = hidden_in_data.detach().cpu().numpy()
            if isinstance(cell_in_data, torch.Tensor):
                cell_in_data = cell_in_data.detach().cpu().numpy()

            output, hidden_out, cell_out = self.session.run(
                None,
                {
                    self.input_name: input_data,
                    self.hidden_in_name: hidden_in_data,
                    self.cell_in_name: cell_in_data,
                },
            )
        else:
            # PyTorch推理路径
            with torch.no_grad():
                if isinstance(input_data, np.ndarray):
                    input_data = torch.from_numpy(input_data).to(self.device)
                if isinstance(hidden_in_data, np.ndarray):
                    hidden_in_data = torch.from_numpy(hidden_in_data).to(self.device)
                if isinstance(cell_in_data, np.ndarray):
                    cell_in_data = torch.from_numpy(cell_in_data).to(self.device)
                output, hidden_out, cell_out = self.model(
                    input_data, hidden_in_data, cell_in_data
                )
        return output, hidden_out, cell_out


def parse_args():
    parser = argparse.ArgumentParser(description="Deployment script.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        help="Run to load from.",
        default=f"{LEGGED_GYM_ROOT_DIR}/logs/U_H1_R/exported/",
    )
    parser.add_argument(
        "--use-onnx", action="store_true", help="Use ONNX runtime for inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for inference",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 初始化部署器
    if args.use_onnx:
        model_path = args.model_path + "policy_lstm_2.onnx"
        # model_path = args.model_path + "policy_lstm.onnx"
    else:
        model_path = args.model_path + "policy_lstm_1.pt"

    deployer = ModelDeployer(
        model_path=model_path, use_onnx=args.use_onnx, device=args.device
    )
    test_input = torch.zeros(1, 44)
    test_hidden_state = torch.zeros(1, 1, 64)
    test_cell_state = torch.zeros(1, 1, 64)
    # 运行推理
    output,_,_ = deployer.predict(test_input, test_hidden_state, test_cell_state)
    print(f"推理完成! 使用 {'ONNX' if args.use_onnx else 'PyTorch'} 运行时")
    print("输出形状:", output.shape)
    print("输出:", output)
