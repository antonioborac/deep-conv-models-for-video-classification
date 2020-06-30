inp_1 = tf.keras.layers.Input(shape=(224,224,3))
inp_2 = tf.keras.layers.Input(shape=(2,))

conv_1 = tf.keras.layers.Conv2D(4,kernel_size=[5,5])(inp_1)
d_1 = tf.keras.layers.Dense(3)(inp_2)

model = tf.keras.Model(inputs=[inp_1,inp_2], outputs=[conv_1,d_1]