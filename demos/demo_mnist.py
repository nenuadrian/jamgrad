from jamgrad import Tensor
from jamgrad.nn import Linear, relu, softmax, cross_entropy_loss
import numpy as np
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


def train_step(model, x_batch, y_batch, lr=0.001):
    predictions = model(x_batch)
    loss = cross_entropy_loss(predictions, y_batch)

    # Zero gradients
    for param in model.parameters():
        param.grad = None

    loss.backward()

    # Update parameters with gradient clipping
    for param in model.parameters():
        if param.grad is not None:
            # Clip gradients to prevent exploding gradients
            grad_clipped = np.clip(param.grad, -1.0, 1.0)
            param.data = param.data - lr * grad_clipped

    return loss.data, predictions


if __name__ == "__main__":
    print("=== Real MNIST Neural Network Demo ===")

    X, y = load_mnist_real()
    print(f"Loaded MNIST subset: {X.shape[0]} samples")

    y_onehot = one_hot_encode(y, 10)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42
    )

    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    model = MNISTNet(input_size=X_train.shape[1])

    batch_size = 32
    epochs = 1000
    lr = 0.01

    print(f"Training for {epochs} epochs with batch size {batch_size}...")

    best_acc = 0
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        for i in range(0, len(X_train), batch_size):
            end_idx = min(i + batch_size, len(X_train))
            x_batch = Tensor(X_train_shuffled[i:end_idx])
            y_batch = Tensor(y_train_shuffled[i:end_idx])

            loss, _ = train_step(model, x_batch, y_batch, lr)
            
            # Handle scalar/array loss
            if hasattr(loss, 'item'):
                total_loss += loss.item()
            elif np.isscalar(loss):
                total_loss += loss
            else:
                total_loss += float(loss)
            num_batches += 1

        avg_loss = total_loss / num_batches

        if epoch % 10 == 0 or epoch == epochs - 1:
            x_test_tensor = Tensor(X_test)
            y_test_tensor = Tensor(y_test)
            test_pred = model(x_test_tensor)
            test_acc = accuracy(test_pred, y_test_tensor)

            if test_acc > best_acc:
                best_acc = test_acc

            print(f"Epoch {epoch:2d}, Loss: {avg_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    print(f"\nBest Test Accuracy: {best_acc:.4f}")

    x_test_tensor = Tensor(X_test[:10])
    y_test_tensor = Tensor(y_test[:10])
    final_pred = model(x_test_tensor)

    print("\nSample predictions (first 10 test samples):")
    for i in range(10):
        pred_class = np.argmax(final_pred.data[i])
        true_class = np.argmax(y_test[i])
        confidence = final_pred.data[i, pred_class]
        status = "✓" if pred_class == true_class else "✗"
        print(
            f"{status} Sample {i}: Predicted {pred_class}, True {true_class}, Confidence {confidence:.3f}"
        )
        status = "✓" if pred_class == true_class else "✗"
        print(
            f"{status} Sample {i}: Predicted {pred_class}, True {true_class}, Confidence {confidence:.3f}"
        )
