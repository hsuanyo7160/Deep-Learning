import numpy as np
import os
import io
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Embedding, Dense, SimpleRNN
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import time

######### Open text files #########
with io.open('shakespeare.txt', 'r') as file:
    text = file.read()
with io.open('shakespeare_valid.txt', 'r', encoding='utf8') as f:
    valid_text = f.read()
vocab = sorted(set(text))
###################################


# Creating a mapping from unique characters to indices
char_to_index = {char: idx for idx, char in enumerate(vocab)}
index_to_char = np.array(vocab)

# Convert text to integers
text_as_int = np.array([char_to_index[c] for c in text])
valid_text_as_int = np.array([char_to_index[c] for c in valid_text])

# Define sequence length and create training examples
seq_length = 100
examples_per_epoch = len(text) // (seq_length + 1)

# Create the dataset by slicing text
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
char_valid_dataset = tf.data.Dataset.from_tensor_slices(valid_text_as_int)
# Create input-target sequences
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)
valid_sequences = char_valid_dataset.batch(seq_length + 1, drop_remainder=True)

# Function to split input and target
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)
valid_dataset = valid_sequences.map(split_input_target)


# Batch size
BATCH_SIZE = 64
BUFFER_SIZE = 10000

# Create the batches
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Length of the vocabulary in chars
embedding_dim = 256  # Dimension of the embedding layer
rnn_units = 1024     # Number of LSTM units

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        SimpleRNN(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        Dense(vocab_size)
    ])
    return model

# 設定檢查點路徑
checkpoint_dir = 'tmpsmalldataset/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

# Build the model
model = build_model(vocab_size=len(vocab), embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=BATCH_SIZE)

# Print the model summary
model.summary()

# Define the loss function
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

# Number of epochs for training
EPOCHS = 40  # Increase this for better results

# Train the model
history = model.fit(dataset, epochs=EPOCHS, validation_data=valid_dataset, callbacks=[checkpoint_callback])

def render_training_history(training_history):
    loss = training_history.history['loss']
    val_loss = training_history.history['val_loss']
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss', linestyle='--')
    plt.legend()
    plt.grid(linestyle='--', linewidth=1, alpha=0.5)
    plt.show()

render_training_history(history)