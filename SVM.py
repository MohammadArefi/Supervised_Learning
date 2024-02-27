import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import metrics

data = pd.read_csv(r'/Users/mohammad/Downloads/Uni/Term 8/bank-sample-test.csv')

X = data.loc[:, data.columns != 'subscribed']
Y = data.loc[:, data.columns == 'subscribed']

x = X.to_numpy()
y = Y.to_numpy().ravel()

training_x, test_x = x[:60,:], x[60:,:]
training_y, test_y = y[:60], y[60:]
svm1 = svm.SVC()
svm1.fit(training_x, training_y)

predicted = svm1.predict(test_x)

accuracy = accuracy_score(test_y, predicted)
print("Prediction Accuracy:", accuracy)

auc = metrics.roc_auc_score(y[60:], predicted)
print('AUC Score Is:', auc)

fpr, tpr, _ = metrics.roc_curve(test_y, predicted)
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
