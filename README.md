# Housing Price Prediction with Gradient Boosting and Random Forest

This repository explores the prediction of housing prices using two powerful machine learning algorithms: **Gradient Boosting Regressor** and **Random Forest Regressor**. The project leverages the **California Housing Dataset**, a popular benchmark dataset for regression problems.

## Motivation

Understanding housing prices is critical for real estate markets, urban planning, and individual decision-making. By leveraging machine learning, this project aims to:

- Analyze key features influencing housing prices.
- Compare the performance of Gradient Boosting and Random Forest models.
- Visualize results through feature importance, residual analysis, and learning curves.

## Objectives

1. **Model Training and Evaluation**:
   - Train Gradient Boosting and Random Forest models.
   - Assess model performance using metrics like:
     - **Mean Squared Error (MSE)**
     - **R-squared (R²)**
     - **Mean Absolute Error (MAE)**

2. **Feature Importance**:
   - Identify the most influential features for each model.

3. **Visualization**:
   - Compare actual vs. predicted values.
   - Analyze residuals and their distributions.
   - Visualize learning curves and partial dependence plots.

4. **Insights**:
   - Highlight patterns in the data.
   - Discuss strengths and limitations of each model.

## Concepts

### Gradient Boosting Regressor

**Gradient Boosting** is a powerful machine learning technique that builds an ensemble of weak learners (typically decision trees) to create a strong predictive model. It does this iteratively, improving the performance of the model at each step.

Gradient Boosting relies on sequential training, where each model corrects the errors of the previous one. It minimizes a specified loss function (such as Mean Squared Error) through gradient descent, making it highly flexible for various regression problems. A critical parameter in Gradient Boosting is the learning rate, which controls the contribution of each tree to the final prediction. Regularization techniques, like limiting tree depth or subsampling, help mitigate overfitting.

Advantages of Gradient Boosting include its ability to handle complex non-linear relationships and robust performance with proper tuning. However, it can overfit if not carefully regularized and is computationally intensive for large datasets.

### Random Forest Regressor

**Random Forest** is an ensemble learning technique that combines multiple decision trees to make accurate and stable predictions. It uses bagging (bootstrap aggregating) to train trees independently on random subsets of the data. At each split, a random subset of features is considered, reducing tree correlation and improving robustness. This parallel approach makes Random Forest highly scalable and faster to train compared to sequential methods like Gradient Boosting.

Random Forest excels at handling high-dimensional data and missing values while offering feature importance measures for interpretability. However, it may struggle to capture complex relationships as effectively as boosting methods and might have less precision for small datasets.

### Conceptual Comparison

| **Aspect**                 | **Gradient Boosting**                                      | **Random Forest**                                      |
|----------------------------|---------------------------------------------------------|-------------------------------------------------------|
| **Training Process**       | Sequential (models correct predecessor errors)          | Parallel (independent training of trees)            |
| **Bias vs. Variance**      | Low bias, high variance (can overfit)                   | Balanced bias and variance (less prone to overfitting) |
| **Model Complexity**       | More sensitive to hyperparameter tuning                 | Generally robust with default settings              |
| **Interpretability**       | More challenging to interpret (depends on loss gradient)| Provides straightforward feature importance          |
| **Computation**            | Slower (sequential training)                            | Faster (parallelizable)                             |

## Visualization

<table>
  <tr>
    <td><img src="https://github.com/capuanomassimo/HousePricing/blob/main/Figure_1.png" alt="Alt text 1" width="400"/></td>
    <td><img src="https://github.com/capuanomassimo/HousePricing/blob/main/Figure_2.png" alt="Alt text 2" width="400"/></td>
    <td><img src="https://github.com/capuanomassimo/HousePricing/blob/main/Figure_3.png" alt="Alt text 3" width="400"/></td>
  </tr>
</table>

<table>
  <tr>
    <td><img src="https://github.com/capuanomassimo/HousePricing/blob/main/Figure_4.png" alt="Alt text 1" width="400"/></td>
    <td><img src="https://github.com/capuanomassimo/HousePricing/blob/main/Figure_5.png" alt="Alt text 2" width="400"/></td>
    <td><img src="https://github.com/capuanomassimo/HousePricing/blob/main/Figure_6.png" alt="Alt text 3" width="400"/></td>
  </tr>
</table>

<table>
  <tr>
    <td><img src="https://github.com/capuanomassimo/HousePricing/blob/main/Figure_7.png" alt="Alt text 1" width="400"/></td>
    <td><img src="https://github.com/capuanomassimo/HousePricing/blob/main/Figure_8.png" alt="Alt text 2" width="400"/></td>
    <td><img src="https://github.com/capuanomassimo/HousePricing/blob/main/Figure_9.png" alt="Alt text 3" width="400"/></td>
  </tr>
</table>

<table>
  <tr>
    <td><img src="https://github.com/capuanomassimo/HousePricing/blob/main/Figure_10.png" alt="Alt text 1" width="400"/></td>
    <td><img src="https://github.com/capuanomassimo/HousePricing/blob/main/Figure_11.png" alt="Alt text 2" width="400"/></td>
    <td><img src="https://github.com/capuanomassimo/HousePricing/blob/main/Figure_12.png" alt="Alt text 3" width="400"/></td>
  </tr>
</table>

<table>
  <tr>
    <td><img src="https://github.com/capuanomassimo/HousePricing/blob/main/Figure_13.png" alt="Alt text 1" width="400"/></td>
    <td><img src="https://github.com/capuanomassimo/HousePricing/blob/main/Figure_14.png" alt="Alt text 2" width="400"/></td>
    <td><img src="https://github.com/capuanomassimo/HousePricing/blob/main/Figure_15.png" alt="Alt text 3" width="400"/></td>
  </tr>
</table>

<table>
  <tr>
    <td><img src="https://github.com/capuanomassimo/HousePricing/blob/main/Figure_16.png" alt="Alt text 1" width="400"/></td>
    <td><img src="https://github.com/capuanomassimo/HousePricing/blob/main/Figure_17.png" alt="Alt text 2" width="400"/></td>
    <td><img src="https://github.com/capuanomassimo/HousePricing/blob/main/Figure_18.png" alt="Alt text 3" width="400"/></td>
  </tr>
</table>

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/housing-price-prediction.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python main.py
   ```

## Future Work

- Extend the analysis with additional algorithms (e.g., XGBoost, LightGBM).
- Perform hyperparameter optimization for both models.
- Explore advanced visualizations like SHAP for feature contributions.

## Conclusion

This project provides a detailed comparison of two popular regression models for housing price prediction. By analyzing performance metrics, visualizing results, and extracting key insights, it offers a solid foundation for understanding and applying machine learning in regression problems.

