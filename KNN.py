import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn import metrics
data = pd.read_csv(r'/Users/mohammad/Downloads/Uni/Term 8/bank-sample-test.csv')

X = data.loc[:, data.columns != 'subscribed']
Y = data.loc[:, data.columns == 'subscribed']
x = X.to_numpy()
y = Y.to_numpy().ravel()
training_x, test_x = x[:60,:], x[60:,:]
training_y, test_y = y[:60], y[60:]

k_values = [i for i in range (1,31)]
scores = []

scaler = StandardScaler()
X = scaler.fit_transform(x)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, x, y, cv=5)
    scores.append(np.mean(score))
print(scores)

knn_model = KNeighborsRegressor(n_neighbors=4)
knn_model.fit(training_x, training_y)
predicted = knn_model.predict(test_x).round()

accuracy = accuracy_score(test_y, predicted)
print("Prediction Accuracy:", accuracy)

auc = metrics.roc_auc_score(y[60:], predicted)
print('AUC Score Is:', auc)

fpr, tpr, _ = metrics.roc_curve(test_y, predicted)
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



