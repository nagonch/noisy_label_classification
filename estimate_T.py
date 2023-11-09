import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings(
    "ignore"
)  # To ignore all warnings that arise here to improve clarity


def filter_cifar_data(X, y, class_labels=[0, 1, 2]):
    mask = np.isin(y, class_labels)
    X_filtered = X[mask]
    y_filtered = y[mask]
    # Remap the labels to {0, 1, 2}
    label_mapping = {original: new for new, original in enumerate(class_labels)}
    y_mapped = np.array([label_mapping[label] for label in y_filtered])
    return X_filtered, y_mapped


def adjust_transition_matrix(
    matrix, diagonal_boost_factor=1.0, off_diagonal_damping_factor=1.0
):
    adjusted_matrix = matrix.copy()
    np.fill_diagonal(
        adjusted_matrix, adjusted_matrix.diagonal() * diagonal_boost_factor
    )
    adjusted_matrix[
        ~np.eye(adjusted_matrix.shape[0], dtype=bool)
    ] *= off_diagonal_damping_factor
    adjusted_matrix /= adjusted_matrix.sum(
        axis=1, keepdims=True
    )  # Ensure rows sum to 1
    return adjusted_matrix


def estimate_transition_matrix_by_max_prob(
    y_pred_probs, num_classes=3, smoothing_factor=1
):
    transition_matrix = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        max_prob_index = np.argmax(y_pred_probs[:, i])
        transition_matrix[i, :] = y_pred_probs[max_prob_index, :]
    # Apply Laplace smoothing
    transition_matrix += smoothing_factor
    # Normalize the transition matrix to ensure that each row sums to 1
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    return transition_matrix


def train_and_evaluate(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    transition_matrix=None,
    num_classes=3,
):
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=100,
        early_stopping=True,
        random_state=1,
        batch_size=1280,
        verbose=True,
    )

    mlp.fit(X_train, y_train)

    calibrated_clf = CalibratedClassifierCV(
        mlp,
        cv="prefit",
        method="sigmoid",
        n_jobs=4,
    )
    calibrated_clf.fit(X_val, y_val)

    y_pred_probs = calibrated_clf.predict_proba(X_test)

    if transition_matrix is not None:
        transition_matrix_inv = np.linalg.inv(transition_matrix)
        corrected_probs = np.dot(y_pred_probs, transition_matrix_inv)
        y_pred = np.argmax(corrected_probs, axis=1)
    else:
        y_pred = mlp.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted"
    )

    return accuracy, precision, recall, f1_score, y_pred, y_pred_probs


def average_performance_over_splits(
    X, y, num_classes=3, num_splits=10, known_transition_matrix=None, dataset_name=""
):
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    estimated_transition_matrices = []

    for _ in range(num_splits):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        if dataset_name == "CIFAR":
            X_train, y_train = filter_cifar_data(X_train, y_train)
            X_val, y_val = filter_cifar_data(X_val, y_val)

        (
            accuracy,
            precision,
            recall,
            f1_score,
            y_pred,
            y_pred_probs,
        ) = train_and_evaluate(
            X_train,
            y_train,
            X_val,
            y_val,
            X_val,
            y_val,
            transition_matrix=known_transition_matrix,
            num_classes=num_classes,
        )

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

        # Estimate the transition matrix
        estimated_matrix = estimate_transition_matrix_by_max_prob(
            y_pred_probs, num_classes=num_classes
        )
        estimated_transition_matrices.append(estimated_matrix)

    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1_score = np.mean(f1_scores)
    avg_estimated_transition_matrix = np.mean(estimated_transition_matrices, axis=0)

    if dataset_name != "CIFAR":
        avg_estimated_transition_matrix = adjust_transition_matrix(
            avg_estimated_transition_matrix
        )

    # Calculate the standard deviations for each metric
    accuracy_std = np.std(accuracies)
    precision_std = np.std(precisions)
    recall_std = np.std(recalls)
    f1_std = np.std(f1_scores)

    return (
        avg_accuracy,
        avg_precision,
        avg_recall,
        avg_f1_score,
        avg_estimated_transition_matrix,
        accuracy_std,
        precision_std,
        recall_std,
        f1_std,
    )


def load_and_preprocess_data(file_path):
    dataset = np.load(file_path)
    X = dataset["Xtr"] / 255.0  # Training features normalized
    y = dataset["Str"]  # Noisy labels for training
    X_test = dataset["Xts"] / 255.0  # Test features normalized
    y_test = dataset["Yts"]  # Clean labels for test
    X_flattened = X.reshape(X.shape[0], -1)
    X_test_flattened = X_test.reshape(X_test.shape[0], -1)

    return X_flattened, y, X_test_flattened, y_test


# Paths to the datasets
file_path_fashionmnist_05 = "datasets/FashionMNIST0.5.npz"
file_path_fashionmnist_06 = "datasets/FashionMNIST0.6.npz"
file_path_cifar = "datasets/CIFAR.npz"

# Load and preprocess each dataset
(
    X_fashion_05,
    y_fashion_05,
    X_test_fashion_05,
    y_test_fashion_05,
) = load_and_preprocess_data(file_path_fashionmnist_05)
(
    X_fashion_06,
    y_fashion_06,
    X_test_fashion_06,
    y_test_fashion_06,
) = load_and_preprocess_data(file_path_fashionmnist_06)
X_cifar, y_cifar, X_test_cifar, y_test_cifar = load_and_preprocess_data(file_path_cifar)

# Define the known transition matrices for FashionMNIST datasets
transition_matrix_fashion_05 = np.array(
    [[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]]
)

transition_matrix_fashion_06 = np.array(
    [[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]]
)

# Evaluate FashionMNIST0.5 with known transition matrix
(
    avg_accuracy_fashion_05,
    avg_precision_fashion_05,
    avg_recall_fashion_05,
    avg_f1_fashion_05,
    estimated_transition_matrix_fashion_05,
    accuracy_std_fashion_05,
    precision_std_fashion_05,
    recall_std_fashion_05,
    f1_std_fashion_05,
) = average_performance_over_splits(
    X_fashion_05,
    y_fashion_05,
    num_classes=3,
    num_splits=10,
    known_transition_matrix=transition_matrix_fashion_05,
    dataset_name="FashionMNIST0.5",
)

# Evaluate FashionMNIST0.6 with known transition matrix
(
    avg_accuracy_fashion_06,
    avg_precision_fashion_06,
    avg_recall_fashion_06,
    avg_f1_fashion_06,
    estimated_transition_matrix_fashion_06,
    accuracy_std_fashion_06,
    precision_std_fashion_06,
    recall_std_fashion_06,
    f1_std_fashion_06,
) = average_performance_over_splits(
    X_fashion_06,
    y_fashion_06,
    num_classes=3,
    num_splits=10,
    known_transition_matrix=transition_matrix_fashion_06,
    dataset_name="FashionMNIST0.6",
)

# Evaluate CIFAR dataset
(
    avg_accuracy_cifar,
    avg_precision_cifar,
    avg_recall_cifar,
    avg_f1_cifar,
    avg_transition_matrix_cifar,
    accuracy_std_cifar,
    precision_std_cifar,
    recall_std_cifar,
    f1_std_cifar,
) = average_performance_over_splits(
    X_cifar, y_cifar, num_classes=3, num_splits=10, dataset_name="CIFAR"
)

# Print the results for each dataset, including standard deviations
print("FashionMNIST0.5 with Known Transition Matrix:")
print(f"Average Accuracy: {avg_accuracy_fashion_05}")
print(f"Accuracy Standard Deviation: {accuracy_std_fashion_05}")
print(f"Average Precision: {avg_precision_fashion_05}")
print(f"Precision Standard Deviation: {precision_std_fashion_05}")
print(f"Average Recall: {avg_recall_fashion_05}")
print(f"Recall Standard Deviation: {recall_std_fashion_05}")
print(f"Average F1 Score: {avg_f1_fashion_05}")
print(f"F1 Score Standard Deviation: {f1_std_fashion_05}")
print(f"Known Transition Matrix:\n{transition_matrix_fashion_05}")
print(f"Estimated Transition Matrix:\n{estimated_transition_matrix_fashion_05}\n")

print("FashionMNIST0.6 with Known Transition Matrix:")
print(f"Average Accuracy: {avg_accuracy_fashion_06}")
print(f"Accuracy Standard Deviation: {accuracy_std_fashion_06}")
print(f"Average Precision: {avg_precision_fashion_06}")
print(f"Precision Standard Deviation: {precision_std_fashion_06}")
print(f"Average Recall: {avg_recall_fashion_06}")
print(f"Recall Standard Deviation: {recall_std_fashion_06}")
print(f"Average F1 Score: {avg_f1_fashion_06}")
print(f"F1 Score Standard Deviation: {f1_std_fashion_06}")
print(f"Known Transition Matrix:\n{transition_matrix_fashion_06}")
print(f"Estimated Transition Matrix:\n{estimated_transition_matrix_fashion_06}\n")

print("CIFAR with Estimated Transition Matrix:")
print(f"Average Accuracy: {avg_accuracy_cifar}")
print(f"Accuracy Standard Deviation: {accuracy_std_cifar}")
print(f"Average Precision: {avg_precision_cifar}")
print(f"Precision Standard Deviation: {precision_std_cifar}")
print(f"Average Recall: {avg_recall_cifar}")
print(f"Recall Standard Deviation: {recall_std_cifar}")
print(f"Average F1 Score: {avg_f1_cifar}")
print(f"F1 Score Standard Deviation: {f1_std_cifar}")
print(f"Estimated Transition Matrix:\n{avg_transition_matrix_cifar}\n")
