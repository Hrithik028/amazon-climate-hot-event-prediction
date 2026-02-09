from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np

def save_accuracy_plot(history, out_path: str, title: str = "Accuracy vs Epochs") -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.plot(history.history.get("accuracy", []), label="train")
    plt.plot(history.history.get("val_accuracy", []), label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_loss_plot(history, out_path: str, title: str = "Loss vs Epochs") -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.plot(history.history.get("loss", []), label="train")
    plt.plot(history.history.get("val_loss", []), label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_confusion_matrix(cm: np.ndarray, out_path: str, title: str = "Confusion Matrix") -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_true_vs_pred(y_true, y_pred, out_path: str, title: str = "True vs Predicted") -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.scatter(y_true, y_pred, s=10)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
