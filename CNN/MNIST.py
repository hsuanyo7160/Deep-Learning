import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LeakyReLU
import matplotlib.pyplot as plt
import numpy as np
import os
# Prevent OpenMP-related error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))  # Reshape to add channel dimension
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))      # Reshape to add channel dimension
x_train = x_train.astype('float32') / 255                  # Normalize to [0, 1]
x_test = x_test.astype('float32') / 255                      # Normalize to [0, 1]

# Convert class vectors to binary class matrices (one-hot encoding)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Construct the CNN model
model = models.Sequential()
#####Layer 1#####
model.add(layers.Conv2D(16, (7,7), activation="relu", padding="same", data_format="channels_last", input_shape=(28,28,1)))
#model.add(layers.Conv2D(64, (5, 5), activation="relu", padding="same",kernel_regularizer=regularizers.l2(0.01),input_shape=(28, 28, 1), data_format="channels_last"))
#model.add(layers.Conv2D(16, (7,7), strides=(3, 3), activation="relu", padding="same", data_format="channels_last", input_shape=(28,28,1)))
model.add(layers.MaxPooling2D(pool_size=(2,2), data_format="channels_last"))

#####Layer 2#####
model.add(layers.Conv2D(36, (7,7), activation="relu", padding="same", data_format="channels_last"))
#model.add(layers.Conv2D(64, (5, 5), activation="relu", padding="same",kernel_regularizer=regularizers.l2(0.01), data_format="channels_last"))
#model.add(layers.Conv2D(36, (7, 7), strides=(3, 3), activation="relu", padding="same", data_format="channels_last"))
model.add(layers.MaxPooling2D(pool_size=(2,2), data_format="channels_last"))

#####Flatten Layer#####
model.add(layers.Flatten())
model.add(layers.Dense(128, activation="relu"))
#model.add(layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation="softmax"))
#model.add(layers.Dense(10, activation="softmax", kernel_regularizer=regularizers.l2(0.01)))


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Lists to store training metrics per iteration and per epoch
train_loss = []
train_accuracy = []
val_accuracy = []
class MetricsLogger(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        train_loss.append(logs['loss'])
        train_accuracy.append(logs['accuracy'])

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy.append(logs['val_accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
#history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1, callbacks=[MetricsLogger()])

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')

# 取得 epoch 為單位的訓練和驗證準確率
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs_range = np.arange(1, len(train_accuracy) + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, train_accuracy, label='Training Accuracy', color='blue')
plt.plot(epochs_range, val_accuracy, label='Validation Accuracy', color='orange')
plt.title('Training and Validation Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

'''
# Plot Accuracy
iterations = np.arange(len(train_loss))
epochs_range = np.arange(1, 5 + 1)
plt.figure(figsize=(14, 6))

# Plot training accuracy per iteration and validation accuracy per epoch
plt.subplot(1, 2, 1)
plt.plot(iterations, train_accuracy, label='Training Accuracy (per iteration)', color='blue')
plt.plot(epochs_range * len(x_train) // 64, val_accuracy, label='Validation Accuracy (per epoch)', color='orange')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid()
# Plot training loss per iteration
plt.subplot(1, 2, 2)
plt.plot(iterations, train_loss, label='Training Loss (per iteration)', color='red')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss per Iteration')
plt.legend()
plt.grid()
plt.show()
'''
# Plot Learning Curve: Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Learning Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Flatten the weights to plot histogram of conv2d weights
conv1_weights = model.get_layer('conv2d').get_weights()[0]
flat_weights = conv1_weights.flatten()
plt.figure(figsize=(10, 5))
plt.hist(flat_weights, bins=50, color='blue', alpha=0.7)
plt.title('Histogram of Conv1 Weights')
plt.xlabel('Weight Values')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# Flatten the weights to plot histogram of conv2d_1 weights
conv1_weights = model.get_layer('conv2d_1').get_weights()[0]
flat_weights = conv1_weights.flatten()
plt.figure(figsize=(10, 5))
plt.hist(flat_weights, bins=50, color='blue', alpha=0.7)
plt.title('Histogram of Conv2 Weights')
plt.xlabel('Weight Values')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# Flatten the weights to plot histogram of dense weights
conv1_weights = model.get_layer('dense').get_weights()[0]
flat_weights = conv1_weights.flatten()
plt.figure(figsize=(10, 5))
plt.hist(flat_weights, bins=50, color='blue', alpha=0.7)
plt.title('Histogram of Dense Weights')
plt.xlabel('Weight Values')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# Flatten the weights to plot histogram of Output weights
conv1_weights = model.get_layer('dense_1').get_weights()[0]
flat_weights = conv1_weights.flatten()
plt.figure(figsize=(10, 5))
plt.hist(flat_weights, bins=50, color='blue', alpha=0.7)
plt.title('Histogram of Output Weights')
plt.xlabel('Weight Values')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# Show examples of correct and incorrect classifications
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

correct_indices = np.where(predicted_labels == true_labels)[0]
incorrect_indices = np.where(predicted_labels != true_labels)[0]
# Show examples of correct and incorrect classifications
plt.figure(figsize=(10, 5))
for i, idx in enumerate(np.random.choice(correct_indices, 5, replace=False)):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {predicted_labels[idx]}, True: {true_labels[idx]}")
    plt.axis('off')
# Show examples of incorrect classifications
for i, idx in enumerate(np.random.choice(incorrect_indices, 5, replace=False)):
    plt.subplot(2, 5, i + 6)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {predicted_labels[idx]}, True: {true_labels[idx]}")
    plt.axis('off')
plt.suptitle("Examples of Correct and Incorrect Classifications")
plt.tight_layout()
plt.show()

# 選擇兩個輸入圖像
input_images = x_test[0:2]  # 使用測試集中的前兩個圖像

# 創建一個模型來提取特徵圖
layer_outputs = [layer.output for layer in model.layers if isinstance(layer, layers.Conv2D)]
feature_map_model = models.Model(inputs=model.input, outputs=layer_outputs)

# 獲取輸入圖像的特徵圖
feature_maps = feature_map_model.predict(input_images)

# 指定要顯示的兩個層（例如，第一層和第二層）
layer_indices = [0, 1]  # 第一層和第二層的索引
num_to_show = 5  # 每層顯示 5 個濾波器

# 繪製指定層的特徵圖
ncols = 5  # 每行顯示的列數

# 對於選定的層，分別顯示兩個圖像的特徵圖
for layer_index in layer_indices:
    feature_map = feature_maps[layer_index]  # 獲取當前層的特徵圖
    num_filters = feature_map.shape[-1]  # 當前層的濾波器數量

    # 確保顯示的濾波器數量不超過當前層的濾波器數量
    num_filters_to_show = min(num_to_show, num_filters)

    # 計算行數
    nrows = 2  # 我們只需要兩行，分別顯示兩個圖像

    plt.figure(figsize=(15, nrows * 2))
    
    for j in range(num_filters_to_show):
        # 繪製第一個圖像的特徵圖
        plt.subplot(nrows, ncols, j + 1)
        plt.imshow(feature_map[0, :, :, j], cmap='viridis')
        plt.axis('off')
        if j == 0:
            plt.title(f'Layer {layer_index + 1} - Image 1')

        # 繪製第二個圖像的特徵圖
        plt.subplot(nrows, ncols, ncols + j + 1)
        plt.imshow(feature_map[1, :, :, j], cmap='viridis')
        plt.axis('off')
        if j == 0:
            plt.title(f'Layer {layer_index + 1} - Image 2')

    plt.tight_layout()
    plt.show()