"""
================================================================================
TASK 2: MACHINE LEARNING PREDICTION MODEL - CRICKET MATCH WINNER PREDICTION
================================================================================

OBJECTIVE:
    Build a machine learning model to predict which team will win a cricket match
    using historical match data.

ALGORITHM SELECTION:
    Two algorithms are tested and compared:
    1. LOGISTIC REGRESSION (Primary Choice)
       - Simplest and most interpretable algorithm
       - Good starting point for binary/multi-class classification
       - Fast training and prediction
       - Less prone to overfitting on small datasets
       
    2. RANDOM FOREST (Secondary Choice)
       - More powerful ensemble method
       - Handles non-linear relationships better
       - Automatically handles feature interactions
       - Better with messy/mixed data types
    
    The best model is selected based on F1 score (weighted average for imbalanced classes)

FEATURE ENGINEERING:
    Input features selected for the model:
    1. team_1 (Categorical) - First team in the match
    2. team_2 (Categorical) - Second team in the match
    3. venue (Categorical) - Match venue location
    4. top_scorer (Categorical) - Player with highest score
    5. top_scorer_runs (Numeric) - Runs scored by top scorer
    
    These features help the model learn patterns like:
    - Team performance at specific venues
    - Impact of top individual performances
    - Team matchup history

TARGET VARIABLE:
    winner - The team that won the match (what we're predicting)

DATA PREPROCESSING:
    - Missing values: Filled with "Unknown" (categorical) or median (numeric)
    - Categorical encoding: One-Hot Encoding for nominal features
    - Numeric scaling: Uses median imputation strategy
    - Train-Test split: 70% training, 30% testing
    
MODEL EVALUATION METRICS:
    - Accuracy: Percentage of correct predictions
    - Precision: True positives / (True positives + False positives)
    - Recall: True positives / (True positives + False negatives)
    - F1 Score: Harmonic mean of precision and recall
    - Confusion Matrix: Visual representation of prediction correctness

================================================================================
"""

import os
import logging
import warnings
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = r"C:\Users\gunja\Downloads\veloria-ml-assignment_1\veloria-tech-ml-intern-assignment"

CSV_FILE = os.path.join(BASE_DIR, "match_data.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_FILE = os.path.join(OUTPUT_DIR, "best_model.pkl")
CONFUSION_PNG = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
FEATURE_PNG = os.path.join(OUTPUT_DIR, "feature_importance.png")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "model_comparison.csv")

logging.basicConfig(
    filename=os.path.join(BASE_DIR, "model.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =====================================================
# LOAD DATA
# =====================================================

print("=" * 60)
print("VELORIA TECH - TASK 2")
print("=" * 60)

df = pd.read_csv(CSV_FILE)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

df.fillna("Unknown", inplace=True)

# =====================================================
# FEATURES - FEATURE SELECTION & ENGINEERING
# =====================================================
# 
# Selected Features Explanation:
# - team_1, team_2: Identify the teams playing (crucial for predictions)
# - venue: Different venues have different conditions that favor certain teams
# - top_scorer: Individual performance indicator
# - top_scorer_runs: Quantitative measure of match performance
#
# These features capture:
#   * Team identity and matchup information
#   * Venue advantage/disadvantage patterns
#   * Individual performance impact
#   * Historical context from past matches
#

X = df[
    [
        "team_1",
        "team_2",
        "venue",
        "top_scorer",
        "top_scorer_runs"
    ]
]

y = df["winner"]

# =====================================================
# TARGET ENCODING
# =====================================================

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# =====================================================
# PREPROCESSING
# =====================================================

categorical_features = [
    "team_1",
    "team_2",
    "venue",
    "top_scorer"
]

numeric_features = [
    "top_scorer_runs"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]
            ),
            categorical_features
        ),
        (
            "numeric",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    )
                ]
            ),
            numeric_features
        )
    ]
)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

test_size = 0.3 if len(df) < 20 else 0.2

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=test_size,
    random_state=42
)

# =====================================================
# LOGISTIC REGRESSION MODEL
# =====================================================
# Logistic Regression is chosen as the primary algorithm because:
#   1. It's simple and interpretable - easy to understand how predictions are made
#   2. Works well for classification tasks (predicting which team wins)
#   3. Less likely to overfit on small datasets (like our 10-match dataset)
#   4. Fast training and prediction times
#   5. Good baseline for comparison with more complex models
#
# Max iterations set to 1000 to ensure convergence on limited data

logistic_model = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)

logistic_model.fit(
    X_train,
    y_train
)

lr_predictions = logistic_model.predict(X_test)

# =====================================================
# RANDOM FOREST MODEL
# =====================================================
# Random Forest is included for comparison because:
#   1. More powerful - can capture complex non-linear patterns
#   2. Ensemble method - combines multiple decision trees for better predictions
#   3. Handles mixed data types (categorical + numeric) naturally
#   4. Provides feature importance rankings
#   5. Good with messy/real-world data
#
# N_estimators=200 provides good balance between accuracy and speed

rf_model = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)

rf_model.fit(
    X_train,
    y_train
)

rf_predictions = rf_model.predict(X_test)

# =====================================================
# EVALUATION FUNCTION
# =====================================================

def evaluate_model(
    model_name,
    y_true,
    predictions
):
    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0
    )

    print(f"\n{model_name}")
    print("-" * 40)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

# =====================================================
# EVALUATION
# =====================================================

lr_results = evaluate_model(
    "Logistic Regression",
    y_test,
    lr_predictions
)

rf_results = evaluate_model(
    "Random Forest",
    y_test,
    rf_predictions
)

# =====================================================
# SAVE COMPARISON TABLE
# =====================================================

results_df = pd.DataFrame(
    [lr_results, rf_results]
)

results_df.to_csv(
    RESULTS_CSV,
    index=False
)

print(
    f"\nModel comparison saved:\n{RESULTS_CSV}"
)

# =====================================================
# BEST MODEL
# =====================================================

best_model = (
    rf_model
    if rf_results["F1 Score"] >= lr_results["F1 Score"]
    else logistic_model
)

joblib.dump(
    best_model,
    MODEL_FILE
)

print(
    f"\nBest model saved:\n{MODEL_FILE}"
)

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_test,
    rf_predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.tight_layout()

plt.savefig(
    CONFUSION_PNG
)

plt.close()

print(
    f"\nConfusion matrix saved:\n{CONFUSION_PNG}"
)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

try:

    forest = rf_model.named_steps["classifier"]

    preprocessor.fit(X)

    feature_names = (
        preprocessor
        .named_transformers_["categorical"]
        .named_steps["encoder"]
        .get_feature_names_out(
            categorical_features
        )
    )

    feature_names = list(feature_names)
    feature_names.append("top_scorer_runs")

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": forest.feature_importances_
        }
    )

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        importance_df["Feature"][:10],
        importance_df["Importance"][:10]
    )

    plt.title("Random Forest Feature Importance")

    plt.tight_layout()

    plt.savefig(
        FEATURE_PNG
    )

    plt.close()

    print(
        f"\nFeature importance saved:\n{FEATURE_PNG}"
    )

except Exception as e:

    print(
        f"\nFeature importance skipped: {e}"
    )

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("\nClassification Report")
print("=" * 60)

print(
    classification_report(
        y_test,
        rf_predictions,
        zero_division=0
    )
)

# =====================================================
# RESULTS SUMMARY & DOCUMENTATION
# =====================================================
print("\n" + "=" * 60)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 60)

print("\nALGORITHM COMPARISON:")
print(f"  Logistic Regression Accuracy: {round(lr_results['Accuracy'], 4)}")
print(f"  Logistic Regression F1 Score: {round(lr_results['F1 Score'], 4)}")
print(f"  Random Forest Accuracy:       {round(rf_results['Accuracy'], 4)}")
print(f"  Random Forest F1 Score:       {round(rf_results['F1 Score'], 4)}")

print("\nBEST MODEL SELECTED:")
best_model_name = (
    "Random Forest"
    if rf_results["F1 Score"] >= lr_results["F1 Score"]
    else "Logistic Regression"
)
print(f"  {best_model_name} (based on highest F1 score)")

print("\nFEATURES USED:")
print("  - team_1 (categorical): First team in the match")
print("  - team_2 (categorical): Second team in the match")
print("  - venue (categorical): Match venue location")
print("  - top_scorer (categorical): Player with highest score")
print("  - top_scorer_runs (numeric): Runs scored by top scorer")

print("\nDATASET INFORMATION:")
print(f"  Total matches: {len(df)}")
print(f"  Training set: {len(X_train)} matches")
print(f"  Testing set: {len(X_test)} matches")

print("\nOUTPUT FILES GENERATED:")
print(f"  - Model file: {MODEL_FILE}")
print(f"  - Confusion matrix plot: {CONFUSION_PNG}")
print(f"  - Model comparison CSV: {RESULTS_CSV}")

print("\n" + "=" * 60)
print("Task 2 Completed Successfully!")
print("=" * 60)