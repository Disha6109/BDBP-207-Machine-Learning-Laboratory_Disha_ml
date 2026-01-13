from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    dataset = fetch_california_housing()
    X = dataset.data
    y = dataset.target
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    return X, y

def divide_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
    return X_train, X_test, y_train, y_test

def standardise_data(X_train,X_test):
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)
    return X_train,X_test


def initialize_model():
    return LinearRegression()


def train_model(model,X_train,y_train):
    model.fit(X_train,y_train)
    return model

def test_model(model,X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2

def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = divide_data(X, y)

    X_train, X_test = standardise_data(X_train, X_test)

    model = initialize_model()

    model = train_model(model, X_train, y_train)

    r2 = test_model(model, X_test, y_test)

    print("R² Score:", r2)

if __name__ == '__main__':
    main()