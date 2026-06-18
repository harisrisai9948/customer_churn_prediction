# src/train.py
from sklearn.model_selection import GridSearchCV

from src.evaluate import evaluate_model
from sklearn.model_selection import cross_val_score

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# --- ADD THESE TWO LINES ---
from sklearn.model_selection import train_test_split



from src.preprocess import build_preprocessor



from sklearn.ensemble import RandomForestClassifier




def load_data(path):
    return pd.read_csv(path)
def clean_data(df):

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["TotalCharges"]
    )

    return df


def prepare_target(df):

    y = df["Churn"]

    X = df.drop(
        ["customerID", "Churn"],
        axis=1
    )

    return X, y
def build_logistic_pipeline(X):


    preprocessor = build_preprocessor(X)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                max_iter=1000,
                random_state=42
            ))
        ]
    )
    return pipeline
def build_random_forest_pipeline(X):

    preprocessor = build_preprocessor(X)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
            )
        ]
    )
    return pipeline

    

   
def main():

    df = load_data("data/customer_churn.csv")
    df = clean_data(df)

    X, y = prepare_target(df)
    logistic_pipeline = build_logistic_pipeline(X)
    param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100]
    }
    grid_search = GridSearchCV(
    estimator=logistic_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
    )
    grid_search.fit(X, y)
    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nBest CV Score:")
    print(grid_search.best_score_)
    rf_pipeline = build_random_forest_pipeline(X)
    logistic_scores = cross_val_score(
    logistic_pipeline,
    X,
    y,
    cv=5,
    scoring="f1_macro"
    ) 
    rf_scores = cross_val_score(
    rf_pipeline,
    X,
    y,
    cv=5,
    scoring="f1_macro"
    )
    print(
    f"\nLogistic Regression CV: {logistic_scores.mean():.4f}"
)

    print(
    f"Random Forest CV: {rf_scores.mean():.4f}"
)
    

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline = logistic_pipeline
    

    pipeline.fit(X_train, y_train)
    joblib.dump(
    pipeline,
    "models/logistic_pipeline.pkl"
     )

    print(
    "Model saved successfully!"
    )
    y_pred = pipeline.predict(X_test)

    y_prob = pipeline.predict_proba(
       X_test
    )[:, 1]
    evaluate_model(
    y_test,
    y_pred,
    y_prob
    )
    

    
   
if __name__ == "__main__":
    main()