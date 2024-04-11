import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
label_names = data['target_names']  
labels = data['target']     
feature_names = data['feature_names'] 
features = data['data']  
train, test, train_labels, test_labels = train_test_split(features, labels,test_size=0.33, random_state=42)

mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='logistic', solver='adam', max_iter=500)

mlp.fit(train,train_labels)


predict_train = mlp.predict(train)

predict_test = mlp.predict(test)

print(accuracy_score(test_labels, predict_test))
