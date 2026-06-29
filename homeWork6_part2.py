import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

##1.) Load dataset with pandas
df = pd.read_csv('Cardiovascular_Disease_Dataset.csv')

##2.) extract features
##data training
##this takes from the first to the 13th coloum
X = df.iloc[:, 1:13].values
y = df.iloc[:, 13].values

##print(len(X))
##print(len(y))
##print(len(X[0]))

##3.) data split
X_train = X[:700]
y_train = y[:700]
X_test = X[700:]
y_test = y[700:]


##4.) train SVM
##train the model
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
##train a 2D model JUST for visualization purposes
X_train_2d = X_train[:, :2]
svm_2d = SVC(kernel='linear')
svm_2d.fit(X_train_2d, y_train)

##5.) predict labels
X_prediction = svm.predict(X_test)
print('X prediction', X_prediction)

##Plot the data
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Positive Heart Problems')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Negative Heart Problems')
plt.scatter(X_test[:, 0], X_test[:, 1], color='green', label='Test Point', marker='X', s=100)

# Plot decision boundary and margins
plt.contour(XX, YY, Z, colors='k', levels=[0], linestyles=['-'])        # decision boundary
plt.contour(XX, YY, Z, colors='k', levels=[-1, 1], linestyles=['--'])   # margins

# Create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 100)
yy = np.linspace(ylim[0], ylim[1], 100)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm_2d.decision_function(xy).reshape(XX.shape)

# Plot decision boundary and margins
plt.contour(XX, YY, Z, colors='k', levels=[0], linestyles=['-'])        # decision boundary
plt.contour(XX, YY, Z, colors='k', levels=[-1, 1], linestyles=['--'])   # margins

##6.) Evaluate the performance
accuracy = accuracy_score(y_test, X_prediction)
precision = precision_score(y_test, X_prediction, average="binary")
recall = recall_score(y_test, X_prediction, average="binary")
f1 = f1_score(y_test, X_prediction, average="binary")
print('accuracy: ', accuracy)
print('precision:  ', precision)
print('recall: ', recall)
print('F1-score: ', f1)

cm = confusion_matrix(y_test, X_prediction)

plt.legend()
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Positive", "Negative"],
    yticklabels=["Negative", "Positive"],
)

plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
plt.show()





