import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('weather_forecast_data.csv'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))



df = pd.read_csv('weather_forecast_data.csv')

sns.pairplot(df, hue="Rain", corner=True)

plt.show()


# Replace categorical values with numerical values
df['Rain'] = df['Rain'].replace({'rain': 1, 'no rain': 0})

# Set style
plt.style.use("seaborn-v0_8-darkgrid")

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Distribution of Numerical Features', fontsize=20, fontweight='bold')

# Define feature names and corresponding titles
features = ['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']
titles = ['Temperature Distribution', 'Humidity Distribution', 
          'Wind Speed Distribution', 'Cloud Cover Distribution', 'Pressure Distribution']

# Plot histograms for numerical features
for ax, feature, title in zip(axes.flat[:5], features, titles):
    sns.histplot(df[feature], kde=True, ax=ax, color='royalblue')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(feature, fontsize=12)
    ax.set_ylabel('Count', fontsize=12)

# Plot countplot for Rain
sns.countplot(data=df, x='Rain', ax=axes[1, 2], palette='coolwarm')
axes[1, 2].set_title('Rain Distribution', fontsize=14, fontweight='bold')
axes[1, 2].set_xlabel('Rain (0 = No, 1 = Yes)', fontsize=12)
axes[1, 2].set_ylabel('Count', fontsize=12)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Show plot
plt.show()



X = df.drop(columns=['Rain'])
y = df['Rain']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
precision = precision_score(y_test, y_pred)
print(f'Precision: {precision:.2f}')
recall = recall_score(y_test, y_pred)
print(f'Recall: {recall:.2f}')
f1 = f1_score(y_test, y_pred)
print(f'F1 Score: {f1:.2f}')
