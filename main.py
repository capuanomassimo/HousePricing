import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import learning_curve
from sklearn.inspection import PartialDependenceDisplay

data = fetch_california_housing()
X, y = data.data, data.target
feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
gbr.fit(X_train, y_train)
gbr_pred = gbr.predict(X_test)

rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

gbr_mse = mean_squared_error(y_test, gbr_pred)
gbr_r2 = r2_score(y_test, gbr_pred)
gbr_mae = mean_absolute_error(y_test, gbr_pred)

rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)

gbr_residuals = y_test - gbr_pred
rf_residuals = y_test - rf_pred

print("Gradient Boosting Metrics:")
print(f"Mean Squared Error (MSE): {gbr_mse:.2f}")
print(f"R-squared (R2): {gbr_r2:.2f}")
print(f"Mean Absolute Error (MAE): {gbr_mae:.2f}")

print("\nRandom Forest Metrics:")
print(f"Mean Squared Error (MSE): {rf_mse:.2f}")
print(f"R-squared (R2): {rf_r2:.2f}")
print(f"Mean Absolute Error (MAE): {rf_mae:.2f}")

plt.figure(figsize=(8, 6))
plt.barh(feature_names, gbr.feature_importances_, color="skyblue", alpha=0.8)
plt.xlabel("Feature Importance")
plt.title("Feature Importance: Gradient Boosting")
plt.show()

plt.figure(figsize=(8, 6))
plt.barh(feature_names, rf.feature_importances_, color="orange", alpha=0.8)
plt.xlabel("Feature Importance")
plt.title("Feature Importance: Random Forest")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_test, gbr_pred, alpha=0.6, color="blue", label="Gradient Boosting")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "--", color="red", label="Ideal Line")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted: Gradient Boosting")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_test, rf_pred, alpha=0.6, color="green", label="Random Forest")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "--", color="red", label="Ideal Line")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted: Random Forest")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(y_test - gbr_pred, bins=20, color="blue", alpha=0.7, label="Gradient Boosting")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Residual Distribution: Gradient Boosting")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(y_test - rf_pred, bins=20, color="green", alpha=0.7, label="Random Forest")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Residual Distribution: Random Forest")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(gbr_pred, y_test - gbr_pred, alpha=0.6, color="blue", label="Gradient Boosting")
plt.axhline(0, color="red", linestyle="--", label="Zero Residual")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted: Gradient Boosting")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(rf_pred, y_test - rf_pred, alpha=0.6, color="green", label="Random Forest")
plt.axhline(0, color="red", linestyle="--", label="Zero Residual")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted: Random Forest")
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.barh(feature_names, gbr.feature_importances_, color="skyblue", alpha=0.7, label="Gradient Boosting")
plt.barh(feature_names, rf.feature_importances_, color="orange", alpha=0.5, label="Random Forest")
plt.xlabel("Feature Importance")
plt.title("Feature Importance: Gradient Boosting vs Random Forest")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_test, gbr_pred, alpha=0.6, color="blue", label="Gradient Boosting")
plt.scatter(y_test, rf_pred, alpha=0.6, color="green", label="Random Forest")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "--", color="red", label="Ideal Line")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted: Gradient Boosting vs Random Forest")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(y_test - gbr_pred, bins=20, color="blue", alpha=0.5, label="Gradient Boosting")
plt.hist(y_test - rf_pred, bins=20, color="green", alpha=0.5, label="Random Forest")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Residual Distribution: Gradient Boosting vs Random Forest")
plt.legend()
plt.show()

def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring="neg_mean_squared_error", 
        train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
    )
    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, label="Training Error", color="blue")
    plt.plot(train_sizes, test_scores_mean, label="Test Error", color="orange")
    plt.xlabel("Training Set Size")
    plt.ylabel("Mean Squared Error")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

plot_learning_curve(gbr, X, y, "Learning Curve: Gradient Boosting")
plot_learning_curve(rf, X, y, "Learning Curve: Random Forest")

residual_stats = pd.DataFrame({
    "Model": ["Gradient Boosting", "Random Forest"],
    "Mean Residual": [np.mean(gbr_residuals), np.mean(rf_residuals)],
    "Std Dev Residual": [np.std(gbr_residuals), np.std(rf_residuals)],
    "Skewness": [pd.Series(gbr_residuals).skew(), pd.Series(rf_residuals).skew()],
    "Kurtosis": [pd.Series(gbr_residuals).kurt(), pd.Series(rf_residuals).kurt()],
})

print(residual_stats)

residual_stats.set_index("Model").plot(kind="bar", figsize=(8, 6))
plt.title("Residual Statistics Comparison")
plt.ylabel("Value")
plt.grid()
plt.show()

features = ["MedInc", "AveRooms", "AveOccup"]
PartialDependenceDisplay.from_estimator(gbr, X, feature_names=feature_names, features=features, kind="average", grid_resolution=20)
plt.suptitle("Partial Dependence Plot: Gradient Boosting", fontsize=14)
plt.tight_layout()
plt.show()

PartialDependenceDisplay.from_estimator(rf, X, feature_names=feature_names, features=features, kind="average", grid_resolution=20)
plt.suptitle("Partial Dependence Plot: Random Forest", fontsize=14)
plt.tight_layout()
plt.show()

gbr_absolute_errors = np.abs(y_test - gbr_pred)
rf_absolute_errors = np.abs(y_test - rf_pred)

plt.figure(figsize=(10, 6))
plt.hist(gbr_absolute_errors, bins=50, alpha=0.6, label="Gradient Boosting", color="blue")
plt.hist(rf_absolute_errors, bins=50, alpha=0.6, label="Random Forest", color="green")
plt.title("Distribution of Absolute Errors")
plt.xlabel("Absolute Error")
plt.ylabel("Frequency")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(gbr_residuals, rf_residuals, alpha=0.6, color="purple")
plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
plt.title("Residual Correlation: Gradient Boosting vs Random Forest")
plt.xlabel("Gradient Boosting Residuals")
plt.ylabel("Random Forest Residuals")
plt.grid()
plt.show()
