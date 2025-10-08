from jamgrad import Tensor
from jamgrad.nn import Linear, relu, softmax, cross_entropy_loss
import numpy as np
import argparse
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_mnist_real():
    print("Downloading MNIST dataset...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = mnist.data, mnist.target.astype(int)

    # Use subset for faster training
    X_subset, _, y_subset, _ = train_test_split(
        X, y, train_size=5000, random_state=42, stratify=y
    )

    # Normalize pixel values
    scaler = StandardScaler()
    X_subset = scaler.fit_transform(X_subset)

    return X_subset, y_subset


def one_hot_encode(y, num_classes):
    encoded = np.zeros((len(y), num_classes))
    for i, label in enumerate(y):
        encoded[i, label] = 1.0
    return encoded


def accuracy(predictions, targets):
    pred_classes = np.argmax(predictions.data, axis=1)
    target_classes = np.argmax(targets.data, axis=1)
    return np.mean(pred_classes == target_classes)


class MNISTNet:
    def __init__(self, input_size=784, hidden1=128, hidden2=64, num_classes=10):
        self.layer1 = Linear(input_size, hidden1)
        self.layer2 = Linear(hidden1, hidden2)
        self.layer3 = Linear(hidden2, num_classes)

        # Set labels for visualization
        self.layer1.weight.set_label("W1")
        self.layer1.bias.set_label("b1")
        self.layer2.weight.set_label("W2")
        self.layer2.bias.set_label("b2")
        self.layer3.weight.set_label("W3")
        self.layer3.bias.set_label("b3")

    def __call__(self, x):
        x = relu(self.layer1(x))
        x = relu(self.layer2(x))
        x = self.layer3(x)
        return softmax(x)

    def parameters(self):
        return (
            self.layer1.parameters()
            + self.layer2.parameters()
            + self.layer3.parameters()
        )


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST Neural Network Training")

    # Dataset parameters
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=5000,
        help="Size of MNIST subset to use (default: 5000)",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Test set proportion (default: 0.2)",
    )

    # Model architecture
    parser.add_argument(
        "--hidden1",
        type=int,
        default=128,
        help="First hidden layer size (default: 128)",
    )
    parser.add_argument(
        "--hidden2", type=int, default=64, help="Second hidden layer size (default: 64)"
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping value (default: 1.0)",
    )

    # Logging and output
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log training progress every N epochs (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--save-graph",
        type=str,
        default=None,
        help='Save computation graph to file (e.g., "mnist_graph")',
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress training progress output"
    )
    parser.add_argument(
        "--sample-predictions",
        type=int,
        default=10,
        help="Number of sample predictions to show (default: 10)",
    )

    return parser.parse_args()


def train_step(model, x_batch, y_batch, lr=0.001, grad_clip=1.0):
    predictions = model(x_batch)
    loss = cross_entropy_loss(predictions, y_batch)

    # Zero gradients
    for param in model.parameters():
        param.grad = None

    loss.backward()

    # Update parameters with gradient clipping
    for param in model.parameters():
        if param.grad is not None:
            grad_clipped = np.clip(param.grad, -grad_clip, grad_clip)
            param.data = param.data - lr * grad_clipped

    return loss.data, predictions


def load_mnist_subset(size):
    """Load a smaller subset of MNIST for faster experimentation."""
    print(f"Downloading MNIST dataset (subset of {size})...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = mnist.data, mnist.target.astype(int)

    X_subset, _, y_subset, _ = train_test_split(
        X, y, train_size=size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_subset = scaler.fit_transform(X_subset)

    return X_subset, y_subset


if __name__ == "__main__":
    args = parse_args()

    # Set random seed
    np.random.seed(args.seed)

    if not args.quiet:
        print("=== MNIST Neural Network Demo ===")
        print(f"Configuration:")
        print(f"  Dataset size: {args.dataset_size}")
        print(f"  Architecture: 784 -> {args.hidden1} -> {args.hidden2} -> 10")
        print(
            f"  Training: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}"
        )

    X, y = (
        load_mnist_real()
        if args.dataset_size >= 5000
        else load_mnist_subset(args.dataset_size)
    )

    if not args.quiet:
        print(f"Loaded MNIST subset: {X.shape[0]} samples")

    y_onehot = one_hot_encode(y, 10)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=args.test_split, random_state=args.seed
    )

    if not args.quiet:
        print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    model = MNISTNet(
        input_size=X_train.shape[1], hidden1=args.hidden1, hidden2=args.hidden2
    )

    if not args.quiet:
        print(f"\nTraining for {args.epochs} epochs...")

    best_acc = 0
    for epoch in range(args.epochs):
        total_loss = 0
        num_batches = 0

        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        for i in range(0, len(X_train), args.batch_size):
            end_idx = min(i + args.batch_size, len(X_train))
            x_batch = Tensor(X_train_shuffled[i:end_idx])
            y_batch = Tensor(y_train_shuffled[i:end_idx])

            loss, _ = train_step(model, x_batch, y_batch, args.lr, args.grad_clip)

            # Handle scalar/array loss
            if hasattr(loss, "item"):
                total_loss += loss.item()
            elif np.isscalar(loss):
                total_loss += loss
            else:
                total_loss += float(loss)
            num_batches += 1

        avg_loss = total_loss / num_batches

        if (
            not args.quiet and epoch % args.log_interval == 0
        ) or epoch == args.epochs - 1:
            x_test_tensor = Tensor(X_test).set_label("test_X")
            y_test_tensor = Tensor(y_test).set_label("test_y")
            test_pred = model(x_test_tensor).set_label("predictions")
            test_acc = accuracy(test_pred, y_test_tensor)

            if test_acc > best_acc:
                best_acc = test_acc

            if not args.quiet:
                print(
                    f"Epoch {epoch:3d}, Loss: {avg_loss:.4f}, Test Accuracy: {test_acc:.4f}"
                )

            # Save graph if requested (only at the end)
            if args.save_graph and epoch == args.epochs - 1:
                if not args.quiet:
                    print(f"\nSaving computation graph to {args.save_graph}.dot...")
                test_pred.save_dot(args.save_graph)

    if not args.quiet:
        print(f"\nBest Test Accuracy: {best_acc:.4f}")

    # Sample predictions
    if args.sample_predictions > 0:
        x_test_tensor = Tensor(X_test[: args.sample_predictions])
        y_test_tensor = Tensor(y_test[: args.sample_predictions])
        final_pred = model(x_test_tensor)
