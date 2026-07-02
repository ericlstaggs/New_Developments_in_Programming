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
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from utilities import visualize_classifier




##1.) Load dataset with pandas via cvs
##df = pd.read_csv('data_random_forests.txt')

input_file = 'data_decision_trees.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]
df = np

##2.) extract features
##data training
# Separate input data into two classes based on labels
class_0 = np.array(X[y==0])
class_1 = np.array(X[y==1])

# Split data into training and testing datasets   ERROR: REMOVED train_test_split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

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

# Visualize input data
plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], s=75, 
        facecolors='black', linewidth=1, marker='x')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white', 
        edgecolors='black', linewidth=1, marker='o')
plt.title('Input data')

##Plot the data
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Positive Heart Problems')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Negative Heart Problems')
plt.scatter(X_test[:, 0], X_test[:, 1], color='green', label='Test Point', marker='X', s=100)

# Decision Trees classifier 
params = {'random_state': 0, 'max_depth': 4}
classifier = DecisionTreeClassifier(**params)
classifier.fit(X_train, y_train)
visualize_classifier(classifier, X_train, y_train, 'Training dataset')

##6.) Evaluate the performance
accuracy = accuracy_score(y_test, X_prediction)
precision = precision_score(y_test, X_prediction, average="binary")
recall = recall_score(y_test, X_prediction, average="binary")
f1 = f1_score(y_test, X_prediction, average="binary")
print('accuracy: ', accuracy)
print('precision:  ', precision)
print('recall: ', recall)
print('F1-score: ', f1)


##creaat confusion matrix graph
cm = confusion_matrix(y_test, X_prediction)

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
