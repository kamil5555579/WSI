import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from SVM import SVM
from sklearn.preprocessing import StandardScaler

def get_data():
    wine_quality = fetch_ucirepo(id=186) 

    X = wine_quality.data.features 
    y = wine_quality.data.targets 

    # Preprocess data
    y = y['quality'].apply(lambda x: 1 if x > 5 else -1)

    X = X.to_numpy()
    y = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def test_svm(kernel='linear', C=1.0, gamma='auto', X_train=None, X_test=None, y_train=None, y_test=None):

    clf = SVM(kernel=kernel, C=C, gamma=gamma)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = np.mean(y_pred == y_test)

    return accuracy

def main():
    results = pd.DataFrame(columns=['kernel', 'C', 'accuracy'])

    kernels = ['linear', 'rbf', 'poly']
    C_values = [1.0, 10.0, 100.0]

    X_train, X_test, y_train, y_test = get_data()
    X_train = X_train[:2000]
    y_train = y_train[:2000]

    for kernel in kernels:
        for C in C_values:
            accuracy = test_svm(kernel=kernel, C=C, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
            results = results._append({'kernel': kernel, 'C': C, 'accuracy': accuracy}, ignore_index=True)

    print(results)

if __name__ == '__main__':
    main()