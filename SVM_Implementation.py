import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Define the dataset
X = np.array([[10,1],[8,1],[1,8],[1,10]])
y = np.array([0,0,1,1])


test = np.array([[2,6]])

# Train the SVM model
svm = SVC(kernel='linear')
svm.fit(X, y)

# Predict the test label
prediction = svm.predict(test)
print('prediction: ',prediction)

# Plot the data points
plt.figure(figsize=(6, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1')
plt.scatter(test[:, 0], test[:, 1], color='green', label='Test Point', marker='X', s=100)

# Plot decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 100)
yy = np.linspace(ylim[0], ylim[1], 100)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm.decision_function(xy).reshape(XX.shape)

# Plot decision boundary and margins
plt.contour(XX, YY, Z, colors='k', levels=[0], linestyles=['-'])        # decision boundary
plt.contour(XX, YY, Z, colors='k', levels=[-1, 1], linestyles=['--'])   # margins

plt.legend()
plt.title(f'SVM Classification - Test Point Predicted as Class {prediction[0]}')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()
