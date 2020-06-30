base_model = BaseModel(weights='imagenet', input_shape=(224,224,3), include_top=False)

model = Sequential()
model.add(TimeDistributed(base_model,input_shape=(frames,224,224,3)))
model.add(TimeDistributed(Flatten()))
model.add(GlobalMaxPooling1D())
model.add(Dense(101, kernel_regularizer=tensorflow.keras.regularizers.l2(0.1)))
model.add(Dense(classes, kernel_regularizer=tensorflow.keras.regularizers.l2(0.1), activation='softmax'))