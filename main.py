import math
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import pandas as pd

# load the mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape the data
x_train = x_train.reshape(len(x_train), 28, 28, 1)
x_test = x_test.reshape(len(x_test), 28, 28, 1)

# define a model
model = tf.keras.Sequential([
    # base
    # block 1
    layers.Conv2D(filters=6, kernel_size=3, strides=1, padding="same", activation="relu",
                  input_shape=[28, 28, 1]),
    layers.MaxPool2D(),

    # head
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),  # use this layer to try and avoid over training
    layers.Dense(10, activation="softmax"),
])

# compile the model
model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

# train the model
hist = model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))

# predict
preds = model.predict(x_test)

# plot some of the data with a bar graph to show what the neural net thinks it is
plt.figure(1)
num = 25
offset = 900
rows = int(math.sqrt(num))
cols = 2 * int(num / rows)

for i in range(0, 2 * num, 2):
    plt.subplot(rows + 1, cols, i + 1)
    plt.imshow(x_test[i + offset])
    plt.subplot(rows + 1, cols, i + 2)
    plt.bar([i for i in range(10)], preds[i + offset])
    plt.xticks([i for i in range(10)])


# plot the loss and accuracy
plt.figure(2)
history = pd.DataFrame(hist.history)
plt.subplot(121)
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(history["loss"], label="Loss")
plt.plot(history["val_loss"], label="Val_Loss")
plt.legend()
plt.tight_layout()


plt.subplot(122)
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(history["accuracy"], label="Accuracy")
plt.plot(history["val_accuracy"], label="Val_Accuracy")
plt.legend()
plt.tight_layout()

plt.show()

