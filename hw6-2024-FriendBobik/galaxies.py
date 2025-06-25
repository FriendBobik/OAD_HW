import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import json
from forest import RandomForestRegressor

def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None

def main():
    data = load_data('sdss_redshift.csv')
    if data is None:
        return

    features = ['u', 'g', 'r', 'i', 'z']
    X = data[features].values
    y = data['redshift'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    rf = RandomForestRegressor(num_trees=25, tree_depth=5)
    rf.fit(X_train, y_train)

    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    plt.figure(figsize=(10, 10))
    plt.scatter(y_train, y_train_pred, label='Train')
    plt.scatter(y_test, y_test_pred, label='Test')
    plt.xlabel('Истинное значение')
    plt.ylabel('Предсказание')
    plt.legend()
    plt.savefig('redshift.png')

    train_std = mean_squared_error(y_train, y_train_pred, squared=False)
    test_std = mean_squared_error(y_test, y_test_pred, squared=False)

    results = {"train": train_std, "test": test_std}
    with open('redshift.json', 'w') as f:
        json.dump(results, f)

    new_data = load_data('sdss.csv')
    if new_data is None:
        return
    
    X_new = new_data[['u', 'g', 'r', 'i', 'z']].values
    y_new_pred = rf.predict(X_new)
    #print(y_new_pred)

    new_data['redshift'] = y_new_pred
    new_data.to_csv('sdss_predict.csv', index=False)

if __name__ == "__main__":
    main()