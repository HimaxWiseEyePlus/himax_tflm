import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_mnist import Net

PYTORCH_PATH = "pytorch_mnist_cnn.pt"
ONNX_PATH = "mnist_cnn.onnx"

# Load the PyTorch model
trained_model = Net()
trained_model.load_state_dict(torch.load(PYTORCH_PATH))

# Export PyTorch model to ONNX model
dummy_input = torch.randn(1, 1, 28, 28, requires_grad=True)
torch.onnx.export(trained_model, dummy_input, ONNX_PATH, verbose=True)