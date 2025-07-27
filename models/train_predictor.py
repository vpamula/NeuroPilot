import torch
import torch.nn as nn
import torch.nn.functional as F

class StatePredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=32, output_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    predictor = StatePredictor()
    
    example_input = torch.randn(1, 5)
    traced = torch.jit.trace(predictor, example_input)
    traced.save("models/model.pt")

    print("Saved TorchScript model to models/model.pt")