data_aug = tensorflow.keras.preprocessing.image.ImageDataGenerator(
    zoom_range=.2,
    horizontal_flip=True,
    rotation_range=6,
    width_shift_range=.15,
    height_shift_range=.15)