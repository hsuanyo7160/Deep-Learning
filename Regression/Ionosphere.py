import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data Preprocessing
# 1. Data loading
data = pd.read_csv('2024_ionosphere_data.csv')

# 2. Rename columns for better understanding
data.columns = [f'feature_{i+1}' for i in range(data.shape[1]-1)] + ['label']

# 3. Map the labels to binary values
data['label'] = data['label'].map({'g': 1, 'b': 0})

# 4. Randomly shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# 5. Split the data into features and target
X = data.drop('label', axis=1).values
y = data['label'].values.reshape(-1, 1)

# 6. Split the data into training and testing sets
num_samples = data.shape[0]
train_size = int(0.75 * num_samples)
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

# 7. Standardize the data
def standardize(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std_replaced = np.where(std == 0, 1, std)  # Avoid division by zero
    X_train_std = (X_train - mean) / std_replaced
    X_test_std = (X_test - mean) / std_replaced
    return X_train_std, X_test_std

X_train, X_test = standardize(X_train, X_test)

# 8. Define activation functions
def relu_activation(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def swish_activation(x):
    return x / (1 + np.exp(-x))

def swish_derivative(x):
    sigmoid = 1 / (1 + np.exp(-x))
    return sigmoid + x * sigmoid * (1 - sigmoid)
# 9. Define the Layer class
class Layer:
    def __init__(self, input_dim, output_dim, activation):
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.biases = np.zeros((1, output_dim))
        self.activation = activation
        
        if activation == 'relu':
            self.activation_func = relu_activation
            self.activation_derivative = relu_derivative
        elif activation == 'sigmoid':
            self.activation_func = sigmoid_activation
            self.activation_derivative = sigmoid_derivative
        elif activation == 'leaky_relu':
            self.activation_func = leaky_relu
            self.activation_derivative = leaky_relu_derivative
        elif activation == 'swish':
            self.activation_func = swish_activation
            self.activation_derivative = swish_derivative
        elif activation is None:
            self.activation_func = None
            self.activation_derivative = None
            
    def forward(self, input_data):
        self.input = np.array(input_data, dtype=np.float64)
        self.z = np.dot(input_data, self.weights) + self.biases
        if self.activation_func is not None:
            self.a = self.activation_func(self.z)
        else:
            self.a = self.z
        return self.a
    
    def backward(self, da, learning_rate):
        if self.activation_derivative is not None:
            dz = da * self.activation_derivative(self.a)
        else:
            dz = da  
        dw = np.dot(self.input.T, dz)
        db = np.sum(dz, axis=0, keepdims=True)
        da_prev = np.dot(dz, self.weights.T)
        
        self.weights -= learning_rate * dw
        self.biases -= learning_rate * db
        
        return da_prev

# 10. Define the NeuralNetwork class
class NeuralNetwork:
    def __init__(self):
        self.layer1 = Layer(input_dim=X_train.shape[1], output_dim=20, activation='relu')  # First hidden layer with 15 neurons
        self.layer2 = Layer(input_dim=20, output_dim=15, activation='swish')  # Second hidden layer with 10 neurons
        self.output_layer = Layer(input_dim=15, output_dim=1, activation='sigmoid')  # Output layer with 1 neuron for binary classification

    def forward(self, input_data):
        self.layer1_output = self.layer1.forward(input_data)
        self.layer2_output = self.layer2.forward(self.layer1_output)
        return self.output_layer.forward(self.layer2_output)

    def backward(self, output_gradient, learning_rate):
        grad_layer2 = self.output_layer.backward(output_gradient, learning_rate)
        grad_layer1 = self.layer2.backward(grad_layer2, learning_rate)
        self.layer1.backward(grad_layer1, learning_rate)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(1, epochs + 1):
            # Forward pass
            y_pred = self.forward(X)
            
            # Calculate loss (Binary Cross Entropy)
            loss = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))  # Small value to prevent log(0)
            loss_gradient = y_pred - y
            
            # Backward pass
            self.backward(loss_gradient, learning_rate)
            
            # Every 100 epochs, print the loss
            if epoch % 100 == 0:
                loss_list.append(loss)
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

    def predict(self, X):
        y_pred = self.forward(X)
        return (y_pred >= 0.5).astype(int)  # Convert probabilities to binary predictions

# 11. Create a NeuralNetwork instance and start training
nn = NeuralNetwork()

# 12. Set hyperparameters
epochs = 1500
learning_rate = 0.008

# 13. Start training
print("Start training...")
loss_list = []
nn.train(X_train, y_train, epochs, learning_rate)

# 14. Predict
print("\nPredicting...")
y_pred_train = nn.predict(X_train)
y_pred_test = nn.predict(X_test)

# 15. Calculate accuracy
train_accuracy = np.mean(y_pred_train == y_train)
test_accuracy = np.mean(y_pred_test == y_test)

# 16. Print accuracy
print(f'\nTraining Accuracy: {train_accuracy:.4f}')
print(f'Testing Accuracy: {test_accuracy:.4f}')

# 14. Draw the learning curve
plt.plot(loss_list, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.xlim(0, len(loss_list) - 1)   # Set x-axis range
plt.xticks(range(0, len(loss_list), 10))  # Set x-axis ticks every 1000
plt.show()