"""Extract frozen grayscale PCA features from CIFAR-10 without test leakage."""
import pickle
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

DATA = Path(__file__).with_name("data") / "cifar-10-batches-py"
OUT = Path(__file__).with_name("outputs")
OUT.mkdir(exist_ok=True)
P = 32


def load_batch(path):
    with path.open("rb") as handle:
        batch = pickle.load(handle, encoding="bytes")
    images = batch[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    return images.mean(axis=1).reshape(len(images), -1), np.asarray(batch[b"labels"])


def main():
    xs, ys = [], []
    for i in range(1, 6):
        x, y = load_batch(DATA / f"data_batch_{i}")
        xs.append(x)
        ys.append(y)
    x_train, y_train = np.concatenate(xs), np.concatenate(ys)
    x_test, y_test = load_batch(DATA / "test_batch")
    mean = x_train.mean(axis=0)
    centered = x_train - mean
    covariance = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = eigh(covariance, subset_by_index=[len(mean) - P, len(mean) - 1])
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    x_train_pca = centered @ eigenvectors
    x_test_pca = (x_test - mean) @ eigenvectors
    target = OUT / "cifar_features.npz"
    np.savez_compressed(target, X_train=x_train_pca, X_test=x_test_pca,
                        y_train=y_train, y_test=y_test,
                        eigenvalues=eigenvalues, eigenvectors=eigenvectors, mean=mean)
    print("shapes", x_train_pca.shape, x_test_pca.shape)
    print("top eigenvalues", eigenvalues[:8])
    print("saved", target)


if __name__ == "__main__":
    main()
