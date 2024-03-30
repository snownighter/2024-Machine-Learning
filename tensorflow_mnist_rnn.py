import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Suppress warnings and set environment variable
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Print dataset information
print("Size: ", np.size(x_train[0]))
print("Split: ", np.size(x_train) / np.size(x_test))
print("Samples (train): ", np.size(x_train))
print("Samples (test): ", np.size(x_test))

# Define RNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(128, activation='relu', input_shape=(28, 28)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile model
opt = tf.keras.optimizers.Adam()
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(x_train, y_train, epochs=5, verbose=1)

# Evaluate model
score = model.evaluate(x_test, y_test)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Plot accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Model accuracy and loss')
plt.ylabel('Accuracy / Loss')
plt.xlabel('Epoch')
plt.legend(['Accuracy', 'Loss'], loc='upper left')

# Save plot
if not os.path.exists('./img'):
    os.makedirs('./img')
plt.savefig('./img/Accuracy_and_Loss (MNIST_RNN).png')
plt.show()
