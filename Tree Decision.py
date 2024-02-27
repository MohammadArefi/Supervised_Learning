import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn import metrics

data = pd.read_csv(r'/Users/mohammad/Downloads/Uni/Term 8/bank-sample-test.csv')

X = data.loc[:, data.columns != 'subscribed']
Y = data.loc[:, data.columns == 'subscribed']

x = X.to_numpy()
y = Y.to_numpy().ravel()
training_x, test_x = x[:60,:], x[60:,:]
training_y, test_y = y[:60], y[60:]

decision_tree = tree.DecisionTreeClassifier()
decision_tree = decision_tree.fit(training_x, training_y)
predicted = decision_tree.predict(test_x)

accuracy = accuracy_score(test_y, predicted)
print("Prediction Accuracy:", accuracy)

auc = metrics.roc_auc_score(y[60:], predicted)
print('AUC Score Is:', auc)

fpr, tpr, _ = metrics.roc_curve(test_y, predicted)
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# fig = plt.figure(figsize=(25,20))
# _ = tree.plot_tree(decision_tree, filled=True)
# fig.savefig("decistion_tree.png")
