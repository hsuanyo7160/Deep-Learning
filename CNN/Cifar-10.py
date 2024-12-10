import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage import zoom

# Prevent OpenMP-related error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255.0  
x_test = x_test.astype('float32') / 255.0    

# One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,             # Randomly rotate images
    width_shift_range=0.1,         # Randomly shift images horizontally
    height_shift_range=0.1,        # Randomly shift images vertically
    shear_range=0.2,               # Randomly apply shearing transformations
    zoom_range=0.2,                # Randomly zoom in on images
    horizontal_flip=True,          # Randomly flip images horizontally
    fill_mode='nearest'            # Replace missing pixels with the nearest pixel
)
# Apply data augmentation to the training data
datagen.fit(x_train)

# Construct the CNN model
model = models.Sequential()
# First
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.3))
# Second
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.5))
# Third
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.5))
# Flatten
model.add(layers.Flatten())
# Fully connected
model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))  # Avoid overfitting   
model.add(layers.Dense(10, activation='softmax'))  # Output layer

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model and obtain history
history = model.fit(x_train, y_train, epochs=20, batch_size=64, validation_split=0.1)   

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')

# Plot the training and validation accuracy per epoch
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs_range = np.arange(1, len(train_accuracy) + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, train_accuracy, label='Training Accuracy', color='blue')
plt.plot(epochs_range, val_accuracy, label='Validation Accuracy', color='orange')
plt.title('Training and Validation Accuracy per Epoch (CIFAR-10)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# Plot training & validation loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
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

# Flatten the weights to plot histogram of conv2d_2 weights
conv1_weights = model.get_layer('conv2d_2').get_weights()[0]
flat_weights = conv1_weights.flatten()
plt.figure(figsize=(10, 5))
plt.hist(flat_weights, bins=50, color='blue', alpha=0.7)
plt.title('Histogram of Conv3 Weights')
plt.xlabel('Weight Values')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# Flatten the weights to plot histogram of conv2d_3 weights
conv1_weights = model.get_layer('conv2d_3').get_weights()[0]
flat_weights = conv1_weights.flatten()
plt.figure(figsize=(10, 5))
plt.hist(flat_weights, bins=50, color='blue', alpha=0.7)
plt.title('Histogram of Conv4 Weights')
plt.xlabel('Weight Values')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# Flatten the weights to plot histogram of conv2d_4 weights
conv1_weights = model.get_layer('conv2d_4').get_weights()[0]
flat_weights = conv1_weights.flatten()
plt.figure(figsize=(10, 5))
plt.hist(flat_weights, bins=50, color='blue', alpha=0.7)
plt.title('Histogram of Conv5 Weights')
plt.xlabel('Weight Values')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# Flatten the weights to plot histogram of conv2d_5 weights
conv1_weights = model.get_layer('conv2d_5').get_weights()[0]
flat_weights = conv1_weights.flatten()
plt.figure(figsize=(10, 5))
plt.hist(flat_weights, bins=50, color='blue', alpha=0.7)
plt.title('Histogram of Conv6 Weights')
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

# CIFAR-10 class labels
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Show examples of correct and incorrect classifications
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

correct_indices = np.where(predicted_labels == true_labels)[0]
incorrect_indices = np.where(predicted_labels != true_labels)[0]

# Create a figure for displaying the results
plt.figure(figsize=(30, 15))

# Show examples of correct classifications
for i, idx in enumerate(np.random.choice(correct_indices, 5, replace=False)):
    plt.subplot(2, 5, i + 1)
    img_resized = zoom(x_test[idx], (8, 8, 1))  # Zoom the image to 256x256
    plt.imshow(img_resized, interpolation='nearest')  # Use 'nearest' to reduce blurring
    plt.title(f"Pred: {class_labels[predicted_labels[idx]]}, True: {class_labels[true_labels[idx]]}")
    plt.axis('off')

# Show examples of incorrect classifications
for i, idx in enumerate(np.random.choice(incorrect_indices, 5, replace=False)):
    plt.subplot(2, 5, i + 6)
    img_resized = zoom(x_test[idx], (8, 8, 1))  # Zoom the image to 256x256
    plt.imshow(img_resized, interpolation='nearest')  # Use 'nearest' to reduce blurring
    plt.title(f"Pred: {class_labels[predicted_labels[idx]]}, True: {class_labels[true_labels[idx]]}")
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
layer_indices = [0, 1, 2, 3, 4, 5, 6]  # 第一層和第二層的索引
num_to_show = 3  # 每層顯示 5 個濾波器

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