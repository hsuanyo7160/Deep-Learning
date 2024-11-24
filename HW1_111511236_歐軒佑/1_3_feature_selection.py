import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data Preprocessing

def training(x):  #x is the column to be dropped
    # 1. Data loading
    data = pd.read_csv('2024_energy_efficiency_data.csv')

    # 1.1 drop the column
    data = data.drop(columns=x)
    # 2. Randomly shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)
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
                    #print(f'Epoch {epoch}, Loss: {loss:.4f}')
                    loss_list.append(loss)

        def predict(self, X):
            return self.forward(X)
    # 8. Create a NeuralNetwork instance and start training
    nn = NeuralNetwork()

    # 9. Set hyperparameters
    epochs = 10000
    learning_rate = 0.011

    # 10. Start training
    loss_list = []
    nn.train(X_train, y_train, epochs, learning_rate)

    # 11. Predict
    y_pred_train = nn.predict(X_train)
    y_pred_test = nn.predict(X_test)

    # 12. Calculate RMSE
    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_pred - y_true) ** 2))
    train_rmse = rmse(y_train, y_pred_train)
    test_rmse = rmse(y_test, y_pred_test)
    return train_rmse,test_rmse

average_train_rmse = []
average_test_rmse = []
data_column = ['# Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area', 'Overall Height', 'Orientation', 'Glazing Area', 'Glazing Area Distribution']
for i in data_column:
    average_train = 0
    average_test = 0
    for j in range(3):  #Average of 3 times
        train_rmse,test_rmse = training(i)
        average_train += train_rmse
        average_test += test_rmse
    average_train_rmse.append(average_train/3)
    average_test_rmse.append(average_test/3)
    print(f'Average Training RMSE after removing {i}: {average_train/3:.4f}')
    print(f'Average Testing RMSE after removing {i}: {average_test/3:.4f}')

# 繪製長條圖
fig, ax = plt.subplots(figsize=(10, 6))

# 設定長條的寬度
bar_width = 0.35
index = range(len(data_column))

# 繪製訓練 RMSE 和測試 RMSE 的長條圖
bar1 = ax.bar(index, average_train_rmse, bar_width, label='Training RMSE', color='b')
bar2 = ax.bar([i + bar_width for i in index], average_test_rmse, bar_width, label='Testing RMSE', color='orange')

# 設置標題和標籤
ax.set_xlabel('Features')
ax.set_ylabel('Average RMSE')
ax.set_title('Average RMSE for Training and Testing')
ax.set_xticks([i + bar_width / 2 for i in index])  # 調整 x 軸標籤的位置
ax.set_xticklabels(data_column)  # 設置 x 軸標籤
ax.legend()

# 顯示圖形
plt.tight_layout()
plt.show()