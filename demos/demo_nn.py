from jamgrad import Tensor
from jamgrad.nn import Linear, relu
import numpy as np
import argparse


def mse_loss(predictions, targets):
    diff = predictions - targets
    squared_diff = diff * diff
    return squared_diff.sum() * (1.0 / predictions.data.size)


class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.layer1 = Linear(input_size, hidden_size)
        self.layer2 = Linear(hidden_size, output_size)

        # Set labels for visualization
        self.layer1.weight.set_label("W1")
        self.layer1.bias.set_label("b1")
        self.layer2.weight.set_label("W2")
        self.layer2.bias.set_label("b2")

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple Neural Network for XOR function"
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs (default: 1000)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.5, help="Learning rate (default: 0.5)"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=4, help="Hidden layer size (default: 4)"
    )

    # Logging and output
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Log training progress every N epochs (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--save-graph",
        type=str,
        default=None,
        help='Save computation graph to file (e.g., "xor_graph")',
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress training progress output"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    np.random.seed(args.seed)

    if not args.quiet:
        print("=== Simple Neural Network Demo ===")
        print(
            f"Configuration: epochs={args.epochs}, lr={args.lr}, hidden_size={args.hidden_size}"
        )

    # XOR dataset
    X = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]], requires_grad=False).set_label("X")
    y = Tensor([[0], [1], [1], [0]], requires_grad=False).set_label("y")

    model = SimpleNN(input_size=2, hidden_size=args.hidden_size, output_size=1)

    if not args.quiet:
        print(f"\nTraining XOR function for {args.epochs} epochs...")

    losses = []
    for epoch in range(args.epochs):
        loss = train_step(model, X, y, lr=args.lr)
        losses.append(loss)

        if not args.quiet and epoch % args.log_interval == 0:
            print(f"Epoch {epoch:4d}, Loss: {loss:.6f}")

    # Final evaluation
    if not args.quiet:
        print(f"\nFinal loss: {losses[-1]:.6f}")
        print("\nFinal predictions:")

    final_predictions = model(X).set_label("predictions")

    for i in range(4):
        pred = final_predictions.data[i, 0]
        target = y.data[i, 0]
        accuracy = "✓" if abs(pred - target) < 0.5 else "✗"
        if not args.quiet:
            print(
                f"{accuracy} Input: {X.data[i]}, Target: {target:.0f}, Prediction: {pred:.4f}"
            )

    # Save computation graph if requested
    if args.save_graph:
        if not args.quiet:
            print(f"\nSaving computation graph to {args.save_graph}.dot...")

        final_predictions.save_dot(args.save_graph)

        if not args.quiet:
            print(
                f"Graph saved! To render: dot -Tpng {args.save_graph}.dot -o {args.save_graph}.png"
            )
