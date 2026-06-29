##import math
##
##p1 = (2,3)
##p2 = (5,7)
##
##distance = math.dist(p1, p2)
##
##print('Distance: ', distance)



import matplotlib.pyplot as plt

x = [2, -2]
y = [3, -3]

plt.scatter(x[0], y[0], color='red', label='point (2,3)', s=100)
plt.scatter(x[1], y[1], color='blue', label='point (-2,-3)', s=100)

plt.xlabel('x')
plt.ylabel('y')

plt.title('plotting point (2,3) a,d (-2,-3)')
plt.axhline(0, color='black', linewidth=.05)
plt.axvline(0, color='black', linewidth=.05)

plt.legend()
plt.grid(True)
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.xticks(range(-5,6,1))
plt.yticks(range(-5,6,1))
plt.show()


