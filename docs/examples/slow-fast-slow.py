def prepare_input(alpha):
    def func(x):
        b = tensorflow.identity(x[:,::alpha])
        return b
    return Lambda(func)
