import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


x = np.array([[1,1],[2,2],[9,9],[10,10]])
y = np.array([0,0,1,1])

test = np.array([[8,8]])
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x,y)
knn_pred = knn.predict(test)

print('Prediction: ', knn_pred)

plt.scatter(x[:,0],x[:,1], c=y, cmap='bwr', edgecolors='k')
plt.show()

