import numpy as np
import tensorflow as tf


images = 'Images/'

train = tf.keras.preprocessing.image_dataset_from_directory(
    images,
    validation_split = 0.2,
    subset='training',
    seed=42,
    image_size=(128,128),
    batch_size=32,
    label_mode="binary",
    shuffle=True,
    interpolation="nearest"
)

test = tf.keras.preprocessing.image_dataset_from_directory(
    images,
    validation_split = 0.2,
    subset='validation',
    seed=42,
    image_size=(128,128),
    batch_size=32,
    label_mode="binary",
    interpolation="nearest",
    shuffle=False
)







model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation="relu",input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
history = model.fit(train,epochs=10)
model.evaluate(test)


model.summary()
model.evaluate(test)