import matplotlib.pyplot as plt
import numpy as np
import math

def func(x):
	return 1./(1.+math.exp(-x))

x = np.arange(-5,5,0.01)
y = [func(x_i) for x_i in x]

plt.plot(x,y)
plt.show()