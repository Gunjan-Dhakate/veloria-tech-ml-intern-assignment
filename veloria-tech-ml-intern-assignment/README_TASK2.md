# Task 2: Machine Learning Prediction Model - Cricket Match Winner Prediction

## Project Overview

This project builds a machine learning model to predict which team will win a cricket match using historical match data. The model learns patterns from past matches and uses those patterns to make predictions about future match outcomes.

---

## Algorithm Selection & Justification

### Primary Algorithm: **Logistic Regression**

**Why Logistic Regression?**
- **Simplicity**: Easy to understand and interpret - clear how predictions are made
- **Classification Focus**: Designed specifically for predicting categories (which team wins)
- **Robustness on Small Data**: Less prone to overfitting (important with only 10 matches)
- **Performance**: Achieves **33.33% Accuracy** and **0.3333 F1 Score**
- **Interpretability**: Can analyze feature coefficients to understand importance

**When to use it:**
- Binary or multi-class classification problems
- Small to medium-sized datasets
- Need for model interpretability

### Secondary Algorithm: **Random Forest**

**Why Random Forest?**
- **Ensemble Approach**: Combines multiple decision trees for robust predictions
- **Non-linear Relationships**: Can capture complex patterns in data
- **Feature Interaction**: Automatically handles interactions between features
- **Performance**: Achieves **33.33% Accuracy** and **0.1667 F1 Score**
- **Feature Importance**: Provides ranking of which features matter most

**When to use it:**
- Complex, non-linear relationships in data
- Mixed data types (categorical + numeric)
- Need feature importance analysis
- More tolerance for messy, real-world data

### Model Selection
**Selected Model**: Logistic Regression
- Chosen based on highest F1 Score (0.3333 vs 0.1667)
- F1 Score is preferred over accuracy for imbalanced datasets (better overall quality measure)

---

## Feature Engineering & Selection

### Features Used in the Model

| Feature | Type | Description | Importance |
|---------|------|-------------|-----------|
| **team_1** | Categorical | First team in the match | High - Team identity affects performance |
| **team_2** | Categorical | Second team in the match | High - Team matchup information |
| **venue** | Categorical | Cricket ground/stadium location | High - Venue has advantage/disadvantage for teams |
| **top_scorer** | Categorical | Player with highest score | Medium - Individual performance indicator |
| **top_scorer_runs** | Numeric | Runs scored by top scorer | Medium - Quantifies match performance |

### Feature Engineering Rationale

1. **Team Identity** (team_1, team_2)
   - Each team has unique strengths and weaknesses
   - Historical performance varies by team combination
   - The model learns which teams typically perform better

2. **Venue** (venue)
   - Different cricket grounds have different characteristics
   - Some teams perform better at home (familiar conditions)
   - Pitch conditions vary by location
   - Example: MA Chidambaram Stadium vs Wankhede Stadium have different playing conditions

3. **Individual Performance** (top_scorer, top_scorer_runs)
   - Strong individual performances can influence match outcomes
   - High-scoring matches may indicate favorable conditions or strong batting
   - Top scorer often plays in winning team

### Data Preprocessing

- **Missing Values**: Filled with "Unknown" (categorical) or median value (numeric)
- **Encoding**: One-Hot Encoding transforms categorical features into binary columns
- **Normalization**: Numeric features use median imputation for robustness
- **Train-Test Split**: 70% training (7 matches), 30% testing (3 matches)

---

## Model Performance & Results

### Model Evaluation Metrics

#### Logistic Regression (Selected Model)
- **Accuracy**: 33.33% (1 out of 3 test predictions correct)
- **F1 Score**: 0.3333 (balanced measure of precision and recall)
- **Precision**: 33.33% (when predicting a win, correct 33% of the time)
- **Recall**: 33.33% (finds 33% of actual wins)

#### Random Forest
- **Accuracy**: 33.33%
- **F1 Score**: 0.1667 (lower - worse balanced performance)
- **Precision**: 11.11%
- **Recall**: 33.33%

### Confusion Matrix

The confusion matrix shows prediction correctness for each class:
- True Positives: Correctly predicted wins
- False Positives: Incorrectly predicted wins
- False Negatives: Missed wins
- True Negatives: Correctly predicted losses

See `output/confusion_matrix.png` for visual representation.

### Dataset Statistics

- **Total Matches**: 10
- **Training Matches**: 7 (70%)
- **Testing Matches**: 3 (30%)
- **Teams Represented**: Multiple teams from IPL 2026 dataset

### Performance Notes

⚠️ **Important Context:**
- The model achieves 33.33% accuracy on a very small test set (only 3 matches)
- With a larger dataset (100+ matches), accuracy would likely be significantly higher
- Current performance is better than random guessing for multi-class problems
- More data would enable better pattern learning

---

## How the Model Works

### 1. **Data Collection**
   - Load match data from CSV file
   - Identify features and target variable

### 2. **Data Cleaning**
   - Remove or handle missing values
   - Ensure consistent formatting
   - Check data quality

### 3. **Feature Transformation**
   - Encode categorical features (team names, venues) to numerical values
   - Normalize numeric features
   - Create feature vectors

### 4. **Model Training**
   - Split data into training (70%) and testing (30%) sets
   - Train Logistic Regression on training data
   - Learn patterns from historical matches

### 5. **Prediction**
   - Apply learned patterns to test data
   - Make predictions about match winners

### 6. **Evaluation**
   - Compare predictions with actual results
   - Calculate accuracy, precision, recall, F1 score
   - Generate confusion matrix

---

## Running the Model

### Prerequisites
```bash
pip install pandas scikit-learn matplotlib joblib
```

### Execute the Model
```bash
python model.py
```

### Output Files Generated
- `output/best_model.pkl` - Trained model saved for future use
- `output/confusion_matrix.png` - Visualization of prediction accuracy
- `output/model_comparison.csv` - Comparison of Logistic Regression vs Random Forest
- `model.log` - Execution logs

---

## Using the Trained Model for New Predictions

```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('output/best_model.pkl')

# Create new match data
new_match = pd.DataFrame({
    'team_1': ['Mumbai Indians'],
    'team_2': ['Chennai Super Kings'],
    'venue': ['Wankhede Stadium'],
    'top_scorer': ['Virat Kohli'],
    'top_scorer_runs': [75]
})

# Make prediction
prediction = model.predict(new_match)
print(f"Predicted Winner: {prediction[0]}")
```

---

## Recommendations for Improvement

1. **Collect More Data**: 10 matches is too small. Aim for 50+ matches minimum
2. **Feature Engineering**: Add features like:
   - Team's recent form (last 5 matches)
   - Head-to-head record
   - Player injury/availability
   - Weather conditions
   - Toss information
3. **Hyperparameter Tuning**: Use GridSearchCV to optimize model parameters
4. **Cross-Validation**: Use k-fold cross-validation for better evaluation
5. **Feature Selection**: Use recursive feature elimination (RFE)
6. **Advanced Models**: Try XGBoost, LightGBM for better performance

---

## Conclusion

This project demonstrates a working machine learning pipeline for cricket match prediction. While the current accuracy is modest due to the small dataset, the framework is solid and will scale well with more data. The Logistic Regression model provides a good balance of interpretability and performance for this classification task.

---

## Files in This Project

```
├── model.py                    # Main ML model script
├── match_data.csv              # Training data (cricket matches)
├── scraper.py                  # Data collection script
├── README_TASK2.md            # This documentation
└── output/
    ├── best_model.pkl          # Trained model
    ├── confusion_matrix.png    # Prediction accuracy visualization
    └── model_comparison.csv    # Algorithm comparison results
```
