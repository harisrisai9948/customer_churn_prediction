import joblib
import pandas as pd
model = joblib.load(
    "models/logistic_pipeline.pkl"
)
feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()
coefficients = model.named_steps[
    "model"
].coef_[0]
importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Coefficient": coefficients
    }
)
importance_df = importance_df.sort_values(
    by="Coefficient",
    ascending=False
)
print("\nTop Churn Drivers:\n")

print(
    importance_df.head(10)
)
print("\nTop Churn Reducers:\n")

print(
    importance_df.tail(10)
)