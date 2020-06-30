model = tf.keras.Sequential

model.add(tf.keras.layers.Dense(15, input_shape=(10,)))
model.add(tf.keras.layers.Dense(8))