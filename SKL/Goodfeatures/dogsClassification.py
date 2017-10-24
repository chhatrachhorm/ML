import numpy as np
import matplotlib.pyplot as plt

# numpy is a mathematics library of python
# matplotlib.pyplot is a graphics library
# types of dogs - population
greyhounds = 500
labs = 500
gray_height = 28 + 4*np.random.randn(greyhounds)
lab_height = 24 + 4*np.random.randn(labs)
print(gray_height)
print(lab_height)

plt.hist(
    [gray_height, lab_height],
    stacked=True, color=['r', 'b'])
plt.show()
# 	Ideal features
# 		• Informative
# 		• Independent
#       • Simple
