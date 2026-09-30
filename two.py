import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("iris.csv")

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
kernel_models = {}

print("Accuracy of SVM Kernels on Iris Dataset:")
for k in kernels:
    svm_clf = SVC(kernel=k, random_state=42)
    svm_clf.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, svm_clf.predict(X_test_scaled))
    kernel_models[k] = (svm_clf, acc)
    print(f"Kernel '{k}': Accuracy = {acc:.4f}")

best_k = max(kernel_models, key=lambda x: kernel_models[x][1])
print(f"\nBest Kernel: '{best_k}' with Accuracy = {kernel_models[best_k][1]:.4f}")

sample_flower = pd.DataFrame({
    'sepal_length': [5.8],
    'sepal_width': [2.8],
    'petal_length': [5.1],
    'petal_width': [2.4]
})
sample_scaled = scaler.transform(sample_flower)
predicted_class = kernel_models[best_k][0].predict(sample_scaled)[0]
print(f"Prediction for input sample: {predicted_class}")
