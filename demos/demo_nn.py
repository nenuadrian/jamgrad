from jamgrad import Tensor
from jamgrad.nn import Linear, relu
import numpy as np


def mse_loss(predictions, targets):
    diff = predictions - targets
    squared_diff = diff * diff
    return squared_diff.sum() * (1.0 / predictions.data.size)


class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.layer1 = Linear(input_size, hidden_size)
        self.layer2 = Linear(hidden_size, output_size)

    def __call__(self, x):
        x = self.layer1(x)
        x = relu(x)
        x = self.layer2(x)
        return x

    def parameters(self):
        return self.layer1.parameters() + self.layer2.parameters()


def train_step(model, x, y, lr=0.01):
    predictions = model(x)
    loss = mse_loss(predictions, y)

    for param in model.parameters():
        param.grad = None

    loss.backward()

    for param in model.parameters():
        if param.grad is not None:
            param.data = param.data - lr * param.grad

    return loss.data


if __name__ == "__main__":
    np.random.seed(42)

    print("=== Simple Neural Network Demo ===")

    X = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]], requires_grad=False)
    y = Tensor([[0], [1], [1], [0]], requires_grad=False)

    model = SimpleNN(input_size=2, hidden_size=4, output_size=1)

    print("Training XOR function...")
    for epoch in range(1000):
        loss = train_step(model, X, y, lr=0.5)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    print("\nFinal predictions:")
    final_predictions = model(X)
    for i in range(4):
        pred = final_predictions.data[i, 0]
        target = y.data[i, 0]
        print(f"Input: {X.data[i]}, Target: {target:.0f}, Prediction: {pred:.4f}")
