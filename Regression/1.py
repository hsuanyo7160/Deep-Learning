import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data Preprocessing
# 1. Data loading
data = pd.read_csv('2024_energy_efficiency_data.csv')

# 2. Randomly shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# 2.1 Feature selection

correlation_matrix = data.corr()
heating_load_correlation = correlation_matrix['Heating Load'].sort_values(ascending=False)
print("Important Feature：")
print(heating_load_correlation)
low_correlation_features = heating_load_correlation[(heating_load_correlation < 0.01) & (heating_load_correlation > -0.01)].index
data = data.drop(columns=low_correlation_features)

# 3. One-Hot Encoding
categorical_cols = ['Orientation', 'Glazing Area Distribution']
for col in categorical_cols: 
    if col not in data.columns:
        categorical_cols.remove(col)            
for col in categorical_cols:                # Check if the columns exist in the dataset
	if col not in data.columns:
		raise ValueError(f"Column '{col}' not found in the dataset")
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)



# 4. Split the data into features and target
num_samples = data.shape[0]
train_size = int(0.75 * num_samples)

train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

X_train = train_data.drop(['Heating Load', 'Cooling Load'], axis=1).values
y_train = train_data[['Heating Load']].values

X_test = test_data.drop(['Heating Load', 'Cooling Load'], axis=1).values
y_test = test_data[['Heating Load']].values

# 5. Standardize the data
def standardize(X_train, X_test):
    X_train = np.array(X_train, dtype=np.float64)  # Convert to numpy array
    X_test = np.array(X_test, dtype=np.float64)    # Convert to numpy array
    
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    
    std_replaced = np.where(std == 0, 1, std)  # Avoid division by zero
    
    X_train_std = (X_train - mean) / std_replaced
    X_test_std = (X_test - mean) / std_replaced
    
    return X_train_std, X_test_std

X_train, X_test = standardize(X_train, X_test)

# 5. Define activation function
def relu_activation(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def linear_activation(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)

def tanh_activation(x):
    x = np.array(x, dtype=np.float64)  # Convert to numpy array
    return np.tanh(x)

def tanh_derivative(x):
    x = np.array(x, dtype=np.float64)  # Convert to numpy array
    return 1 - np.tanh(x)**2

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)
# 6. Define the Layer class
class Layer:
    def __init__(self, input_dim, output_dim, activation):
        # Initialize weights and biases
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.biases = np.zeros((1, output_dim))
        self.activation = activation
        
        # Set activation function and derivative
        if activation == 'relu':
            self.activation_func = relu_activation
            self.activation_derivative = relu_derivative
        elif activation == 'tanh':
            self.activation_func = tanh_activation
            self.activation_derivative = tanh_derivative
        elif activation == 'linear':
            self.activation_func = linear_activation
            self.activation_derivative = linear_derivative
        elif activation == 'leaky_relu':
            self.activation_func = leaky_relu
            self.activation_derivative = leaky_relu_derivative
        elif activation is None:
            self.activation_func = None
            self.activation_derivative = None
    # Forward pass    
    def forward(self, input_data):
        self.input = np.array(input_data, dtype=np.float64)
        self.z = np.dot(input_data, self.weights) + self.biases
        if self.activation_func is not None:
            self.a = self.activation_func(self.z)
        else:
            self.a = self.z
        return self.a
    # Backward pass
    def backward(self, da, learning_rate):
        if self.activation_derivative is not None:
            dz = da * self.activation_derivative(self.z)
        else:
            dz = da  
        # Calculate gradients
        dw = np.dot(self.input.T, dz)
        db = np.sum(dz, axis=0, keepdims=True)
        da_prev = np.dot(dz, self.weights.T)
        
        # Update weights and biases
        self.weights -= learning_rate * dw
        self.biases -= learning_rate * db
        
        return da_prev
# 7. Define the NeuralNetwork class
class NeuralNetwork:
    def __init__(self):
        # Define input layer to hidden layer 1
        self.layer1 = Layer(input_dim=X_train.shape[1], output_dim=15, activation='relu')  # First hidden layer with 15 neurons, using ReLU activation function
        # Define hidden layer 1 to hidden layer 2
        self.layer2 = Layer(input_dim=15, output_dim=10, activation='leaky_relu')   # Second hidden layer with 10 neurons, using leaky_relu activation function
        # Define hidden layer 2 to output layer
        self.output_layer = Layer(input_dim=10, output_dim=1, activation=None)  # 1 output neuron, using linear activation function

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
            
            # Calculate loss and loss gradient
            # MSE: (1/m) * Σ(y_pred - y)^2
            # Derivative: (2/m) * (y_pred - y)
            m = y.shape[0]
            loss = np.sum((y_pred - y) ** 2)
            loss_gradient = (2/m) * (y_pred - y)
            # Backward pass
            self.backward(loss_gradient, learning_rate)
            
            # Every 100 epochs, print the loss
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}')
                loss_list.append(loss)

    def predict(self, X):
        return self.forward(X)
# 8. Create a NeuralNetwork instance and start training
nn = NeuralNetwork()

# 9. Set hyperparameters
epochs = 10000
learning_rate = 0.011

# 10. Start training
print("Start training...")
loss_list = []
nn.train(X_train, y_train, epochs, learning_rate)

# 11. Predict
print("\nPredicting...")
y_pred_train = nn.predict(X_train)
y_pred_test = nn.predict(X_test)

# 12. Calculate RMSE
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))
train_rmse = rmse(y_train, y_pred_train)
test_rmse = rmse(y_test, y_pred_test)

# 13. Print RMSE
print(f'\nTraining RMSE: {train_rmse:.4f}')
print(f'Testing RMSE: {test_rmse:.4f}')


# 14. Draw the learning curve
plt.plot(loss_list, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.xlim(0, len(loss_list) - 1)   # Set x-axis range
plt.xticks(range(0, len(loss_list), 10))  # Set x-axis ticks every 1000
plt.show()


n_samples_to_plot = 50  #  Only plot the first 50 samples
indices = range(n_samples_to_plot)
# 15. Draw the comparison between the predicted value and the actual value in the training set
plt.figure(figsize=(10, 6))
plt.plot(indices, y_train[:n_samples_to_plot], label='label', color='blue', linewidth=2)
plt.plot(indices, y_pred_train[:n_samples_to_plot], label='Predict', color='orange', linewidth=2)
plt.title('Prediction for Training Data (First 25 Samples)')
plt.xlabel('Sample Index')
plt.ylabel('Heating Load (kWh/m²)')
plt.xticks(indices, rotation=45) 
plt.legend()
plt.grid()
plt.show()

# 16. Draw the comparison between the predicted value and the actual value in the test set
plt.figure(figsize=(10, 6))
plt.plot(indices, y_test[:n_samples_to_plot], label='label', color='blue', linewidth=2)
plt.plot(indices, y_pred_test[:n_samples_to_plot], label='Predict', color='red', linewidth=2)
plt.title('Prediction for Testing Data (First 25 Samples)')
plt.xlabel('Sample Index')
plt.ylabel('Heating Load (kWh/m²)')
plt.xticks(indices, rotation=45)  
plt.legend()
plt.grid()
plt.show()