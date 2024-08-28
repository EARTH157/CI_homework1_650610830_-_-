import numpy as np
import matplotlib.pyplot as plt

def activation_function(x, func='sigmoid'):
    if func == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif func == 'relu':
        return np.maximum(0, x)
    elif func == 'tanh':
        return np.tanh(x)
    elif func == 'linear':
        return x
    else:
        raise ValueError("Unsupported activation function")

def activation_derivative(x, func='sigmoid'):
    if func == 'sigmoid':
        return x * (1 - x)
    elif func == 'relu':
        return np.where(x > 0, 1, 0)
    elif func == 'tanh':
        return 1 - x**2
    elif func == 'linear':
        return np.ones_like(x)
    else:
        raise ValueError("Unsupported activation function")

def forward_pass(X, weights, activation_func='sigmoid'):
    layers = [X]
    for w in weights:
        layers.append(activation_function(np.dot(layers[-1], w), func=activation_func))
    return layers

def backpropagation(layers, weights, y, learning_rate, momentum, previous_weights, activation_func='sigmoid'):
    deltas = [y - layers[-1]]
    for i in range(len(layers) - 2, 0, -1):
        deltas.append(deltas[-1].dot(weights[i].T) * activation_derivative(layers[i], func=activation_func))
    deltas.reverse()
    
    new_weights = []
    for i in range(len(weights)):
        weight_update = learning_rate * layers[i].T.dot(deltas[i])
        if previous_weights is None:
            new_weights.append(weights[i] + weight_update)
        else:
            new_weights.append(weights[i] + weight_update + momentum * (weights[i] - previous_weights[i]))
    
    return new_weights

def train_mlp_regression(X, y, hidden_layers, learning_rate, momentum, epochs, activation_func='sigmoid'):
    weights = [np.random.rand(X.shape[1], hidden_layers[0])]
    for i in range(1, len(hidden_layers)):
        weights.append(np.random.rand(hidden_layers[i-1], hidden_layers[i]))
    weights.append(np.random.rand(hidden_layers[-1], y.shape[1]))

    previous_weights = [np.zeros_like(w) for w in weights] 
    errors = []

    for epoch in range(epochs):
        layers = forward_pass(X, weights, activation_func)
        weights = backpropagation(layers, weights, y, learning_rate, momentum, previous_weights, activation_func)
        
        error = np.mean((y - layers[-1]) ** 2)
        errors.append(error)
        print(f"Flood Dataset - Epoch {epoch + 1}/{epochs}, Error: {error}")

        previous_weights = weights.copy()

    plt.figure(figsize=(10, 5))
    plt.plot(errors, label='MSE')
    plt.title(f'Flood Dataset Training Error Over Epochs\n'
              f'Learning Rate: {learning_rate}, Momentum: {momentum}, Epochs: {epochs}')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.legend()
    plt.show()

    final_mse = errors[-1]
    print(f'Final Mean Squared Error: {final_mse}')

    return weights

def train_mlp_classification(X, y, hidden_layers, learning_rate, momentum, epochs, activation_func='sigmoid'):
    # Initialize weights
    weights = [np.random.rand(X.shape[1], hidden_layers[0])]
    for i in range(1, len(hidden_layers)):
        weights.append(np.random.rand(hidden_layers[i-1], hidden_layers[i]))
    weights.append(np.random.rand(hidden_layers[-1], y.shape[1]))

    previous_weights = [np.zeros_like(w) for w in weights]

    errors = []

    for epoch in range(epochs):
        layers = forward_pass(X, weights, activation_func)
        weights = backpropagation(layers, weights, y, learning_rate, momentum, previous_weights, activation_func)
        
        error = -np.mean(y * np.log(layers[-1] + 1e-8) + (1 - y) * np.log(1 - layers[-1] + 1e-8))
        errors.append(error)
        print(f"Cross.pat Dataset - Epoch {epoch + 1}/{epochs}, Error: {error}")

        previous_weights = weights.copy()

    return weights

def predict(X, weights, activation_func='sigmoid'):
    layers = forward_pass(X, weights, activation_func)
    return layers[-1]

def plot_confusion_matrix(y_true, y_pred, fold_idx=None):
    cm = np.zeros((2, 2), dtype=int)
    for i in range(len(y_true)):
        actual = np.argmax(y_true[i])
        predicted = np.argmax(y_pred[i])
        cm[actual, predicted] += 1

    diagonal_sum = np.trace(cm)
    percentage_diagonal = (diagonal_sum / len(y_true)) * 100

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    if fold_idx is not None:
        plt.title(f'Confusion Matrix for Fold {fold_idx}\nDiagonal Percentage: {percentage_diagonal:.2f}%')
    else:
        plt.title(f'Confusion Matrix\nDiagonal Percentage: {percentage_diagonal:.2f}%')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Class 0', 'Class 1'], rotation=45)
    plt.yticks(tick_marks, ['Class 0', 'Class 1'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    for i in range(2):
        for j in range(2):
            plt.text(j, i, f'{cm[i, j]}', horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.grid(False)
    plt.show()

def k_fold_cross_validation(X, y, k, hidden_layers, learning_rate, momentum, epochs, activation_func='sigmoid'):
    fold_size = len(X) // k
    all_predictions = []
    all_true_labels = []
    fold_accuracies = []

    plt.figure(figsize=(12, 6))

    for i in range(k):
        start, end = i * fold_size, (i + 1) * fold_size
        X_test, y_test = X[start:end], y[start:end]
        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)

        weights = train_mlp_classification(X_train, y_train, hidden_layers, learning_rate, momentum, epochs, activation_func)

        predictions = predict(X_test, weights, activation_func)
        all_predictions.extend(predictions)
        all_true_labels.extend(y_test)

        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(y_test, axis=1)
        accuracy = np.mean(predicted_labels == true_labels)
        fold_accuracies.append(accuracy)
        print(f"Fold {i + 1} Accuracy: {accuracy:.2f}")

        plot_confusion_matrix(y_test, predictions, fold_idx=i + 1)

    mean_accuracy = np.mean(fold_accuracies)
    final_accuracy = fold_accuracies[-1]
    print(f"Mean Accuracy: {mean_accuracy:.2f}")
    print(f"Final Accuracy: {final_accuracy:.2f}")

    plt.plot(range(1, k + 1), fold_accuracies, marker='o', linestyle='-', color='b', label='Fold Accuracy')
    plt.axhline(y=mean_accuracy, color='r', linestyle='--', label='Mean Accuracy')
    plt.axhline(y=final_accuracy, color='g', linestyle='-.', label='Final Accuracy')
    plt.title('K-Fold Cross-Validation Accuracy')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    return np.array(all_true_labels), np.array(all_predictions)

def plot_regression_results(desired_output, test_output, sample_size=50):
    indices = np.random.choice(len(desired_output), sample_size, replace=False)
    sampled_desired_output = desired_output[indices]
    sampled_test_output = test_output[indices]

    plt.figure(figsize=(10, 5))
    plt.plot(sampled_desired_output, label='Desired Output', color='blue')
    plt.plot(sampled_test_output, label='Test Output', color='red', linestyle='--')
    plt.title('Desired Output vs Test Output (Flood Dataset)')
    plt.xlabel('Sample')
    plt.ylabel('Water Level')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_classification_results(desired_output, test_output, sample_size=50):
    predicted_labels = np.argmax(test_output, axis=1)
    true_labels = np.argmax(desired_output, axis=1)

    indices = np.random.choice(len(true_labels), sample_size, replace=False)
    sampled_true_labels = true_labels[indices]
    sampled_predicted_labels = predicted_labels[indices]

    plt.figure(figsize=(10, 5))
    plt.plot(sampled_true_labels, label='True Labels', color='blue')
    plt.plot(sampled_predicted_labels, label='Predicted Labels', color='red', linestyle='--')
    plt.title('True Labels vs Predicted Labels (Cross.pat Dataset)')
    plt.xlabel('Sample')
    plt.ylabel('Class')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def read_data(data_type = "regression"):
    if data_type == "regression":
        return read_data_from_flood_dataset('c:/Users/earth/OneDrive/เดสก์ท็อป/Quiz1/Flood_dataset.txt')
    elif data_type == "classification":
        return read_data_from_cross_pat('c:/Users/earth/OneDrive/เดสก์ท็อป/Quiz1/cross.pat.txt')

def read_data_from_flood_dataset(filename):
    data = []
    with open(filename) as f:
        for line in f.readlines()[2:]:
            data.append([float(element) for element in line.split()])
    data = np.array(data)
    np.random.shuffle(data)
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    epsilon = 1e-8
    data = (data - min_vals) / (max_vals - min_vals + epsilon)
    input_data = data[:, :-1]
    design_output = data[:, -1].reshape(-1, 1)
    return input_data, design_output

def read_data_from_cross_pat(filename):
    data = []
    with open(filename) as f:
        a = f.readlines()
        for line in range(1, len(a), 3):
            z = np.array([float(element) for element in a[line].split()])
            zz = np.array([float(element) for element in a[line+1].split()])
            data.append(np.append(z, zz))
    data = np.array(data)
    np.random.shuffle(data)
    input_data = data[:, :-2]
    design_output = data[:, -2:]
    return input_data, design_output

# โหลดข้อมูล Flood Dataset
input_flood, design_output_flood = read_data("regression")

# โหลดข้อมูล Cross.pat Dataset
input_cross, design_output_cross = read_data("classification")

# ตั้งค่าต่างๆ สำหรับการฝึก MLP
hidden_layers_for_flood = [5]  # สามารถปรับเปลี่ยนได้
learning_rate_for_flood = 0.001
momentum_for_flood = 0.9
epochs_for_flood = 50000
activation_func_for_flood = 'sigmoid'  # เปลี่ยนฟังก์ชัน activation

hidden_layers_for_cross_pat = [10, 10]  # สามารถปรับเปลี่ยนได้
learning_rate_for_cross_pat = 0.01
momentum_for_cross_pat = 0.9
epochs_for_cross_pat = 10000
activation_func_for_cross_pat = 'sigmoid'  # เปลี่ยนฟังก์ชัน activation

# ใช้ k-fold cross-validation
k = 10

# ฝึก MLP สำหรับ Flood Dataset
weights_flood = train_mlp_regression(np.array(input_flood), 
                                     np.array(design_output_flood),
                                     hidden_layers_for_flood, 
                                     learning_rate_for_flood, 
                                     momentum_for_flood, 
                                     epochs_for_flood, 
                                     activation_func_for_flood)
predicted_flood = predict(input_flood, weights_flood, activation_func_for_flood)
plot_regression_results(design_output_flood, predicted_flood, sample_size=30)

# ฝึก MLP สำหรับ Cross.pat Dataset และใช้ k-fold cross-validation
true_labels_cross, predictions_cross = k_fold_cross_validation(input_cross, 
                                                               design_output_cross, 
                                                               k, 
                                                               hidden_layers_for_cross_pat, 
                                                               learning_rate_for_cross_pat, 
                                                               momentum_for_cross_pat, 
                                                               epochs_for_cross_pat, 
                                                               activation_func_for_cross_pat)
plot_classification_results(true_labels_cross, predictions_cross, sample_size=30)
plot_confusion_matrix(true_labels_cross, predictions_cross)