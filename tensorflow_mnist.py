import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

#warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print("Size: ", np.size(x_train[0]))
print("Split: ", np.size(x_train)/np.size(x_test))
print("Samples (train): ", np.size(x_train))
print("Samples (test): ", np.size(x_test))


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

opt = tf.keras.optimizers.Adam()
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#tensorboard = tf.keras.callbacks.TensorBoard(log_dir="./logs")

history = model.fit(x_train, y_train, epochs=5, verbose=1)
score = model.evaluate(x_test, y_test)

print("Score: ", score)

plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('model accuracy')
plt.ylabel('acc & loss')
plt.xlabel('epoch')
plt.legend(['acc', 'loss'], loc='upper left')

if not os.path.exists('./img'):
    os.makedirs('./img')
plt.savefig('./img/Accuracy and Loss.png')
plt.show() # after 'savefig'
plt.close()