import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


data = load_breast_cancer()
label_names = data['target_names']  ## malignant, benign
labels = data['target']      ## 0, 1
feature_names = data['feature_names'] ## attributes used for predicting
features = data['data']  ## the mean of the attributes for each case
train, test, train_labels, test_labels = train_test_split(features, labels,test_size=0.33,random_state=42)
## This function splits the data in train and test sets. The test_sizes=0.33 means that 33% of data is used
## for test (and 67% for training). The random_seed = 42 is meant to get always the same train/test set so 
## the experiment has more consistency.

gnb = GaussianNB()
model = gnb.fit(train, train_labels)
preds = gnb.predict(test)
print(accuracy_score(test_labels, preds))