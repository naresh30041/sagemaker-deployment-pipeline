# train.py
import argparse
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")  # SageMaker default model directory
    args = parser.parse_args()

    # Load and split dataset
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

    # Train a RandomForest model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Save the model to the specified model directory
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
