import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 100)
y1 = np.sin(x)
y2 = np.log(x)
y3 = 1/(1+np.exp(-x))

##sin(x)
plt.plot(x, y1, 'r--', label='sin(x)')
##log(x)
plt.plot(x, y2, 'g-', label='log(x)')
##1/(1+e^-x)
plt.plot(x, y3, 'co-', label='f(x)')

plt.axhline(0, color='black', linewidth=.05)
plt.axvline(0, color='red', linewidth=.05)


plt.title('sin(x), log(x), 1/(1+e^-x)')
plt.legend()
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')

plt.xlim(-5,5)
plt.ylim(-5,5)
plt.show()
