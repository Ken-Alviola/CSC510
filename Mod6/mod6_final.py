#!/usr/bin/env python
# coding: utf-8



from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report
import pandas as pd
import matplotlib.pyplot as plt

input("This script creates a dataset using sklearn's make_classification function, splits the data into train and test, and uses the Gaussian Naive Bayes classifier to predict the label of each test datapoint. Press ENTER to continue: ")
print()

# Create dataset"
X, y = make_classification(
    n_features=7,
    n_classes=3,
    n_samples=2500,
    n_informative=3,
    random_state=45,
    n_clusters_per_class=1,
)


# Print dataset characteristics
n_samples= X.shape[0]
n_features = X.shape[1]
n_classes = pd.DataFrame(y).nunique()

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)
print()

# Display data as 3d scatterplot
fig = plt.figure(figsize=(10, 10))
plt.style.use('ggplot')
ax = fig.add_subplot(projection='3d')

sequence_containing_x_vals = X[:, 0]
sequence_containing_y_vals = X[:, 1]
sequence_containing_z_vals = X[:, 2]

ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals, c=y, marker ="*", cmap='plasma')
plt.show()

# Frequency and likelihood tables
df = pd.DataFrame(X)
df['Class']= y

freq_table = df['Class'].value_counts().sort_index()
print("Frequency table: ")
print(freq_table)
print()

print("Likelihood: ")
print(f'Class 0: {freq_table[0]/len(df)*100}%')
print(f'Class 1: {freq_table[1]/len(df)*100}%')
print(f'Class 2: {freq_table[2]/len(df)*100}%')
print()

# Splitting data to train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=45)


# Build a Gaussian Classifier
model = GaussianNB()

# Model training
model.fit(X_train, y_train)

# Predictions on test data
y_pred = model.predict(X_test)


# Display confusion matrix
labels = [0,1,2]
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.grid(False)
plt.show()


# Performance metrics
print("Number of mislabeled points out of a total %d points : %d" 
      %(X_test.shape[0], (y_test != y_pred).sum()))
print()
print(classification_report(y_test, y_pred))




# Model probabilities
predictions = model.predict_proba(X_test)
pred_To_Class_map = pd.DataFrame(predictions*100, columns=['Class 0','Class 1', 'Class 2'])

# Loop for displaying rows from probability dataframe
row = 0
while row != 999:
    row = int(input("Enter the record number 0-824 you would like to see model prediction probabilities of or type 999 to quit: "))
    if row != 999:
        print()
        print(f"Posterior probabilities for point {row}: ")
        print (pred_To_Class_map.iloc[row])
        print()
    else:
        print()
        break

print("Thank you this concludes the demo.")







