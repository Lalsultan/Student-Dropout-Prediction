# Student Dropout Prediction Using Machine Learning

A collaborative machine-learning project exploring whether student demographic, attendance, support, and academic-performance features can predict a binary dropout outcome. The notebook includes exploratory data analysis, categorical encoding, feature engineering, class-imbalance handling with SMOTE, and comparison of Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM, Random Forest, Gradient Boosting, and XGBoost.

## Dataset

The submitted CSV is intentionally not included while its license, provenance, and redistribution terms are checked. The report references a Kaggle-hosted student dropout dataset; obtain it from an authorized source and confirm that its terms permit the intended use. Place the authorized file at `data/raw/student dropout.csv`.

## Setup

Use Python 3.10 or 3.11, create a virtual environment, and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Open `student_dropout_prediction.ipynb` for the project workflow and `src/evaluate_models.py` for a more rigorous baseline. The baseline fits preprocessing within each training fold, chooses a model using training-only cross-validation, and evaluates the selected model once on a held-out test set. It uses class weighting rather than SMOTE. The script has only been smoke-tested with synthetic data; no performance metrics have been reproduced on the project dataset. The CSV is not included, and its license/provenance needs confirmation before public release.

## Important evaluation caveats

The original notebook reported perfect scores for several tree-based models. These figures should **not** be presented as verified performance. Static review identified issues in the submitted version:

- `StandardScaler` was fitted on the full dataset before the train/test split. This portfolio draft now fits it on the training split only.
- `Final_Grade` was included as a predictor. This draft now excludes it until its availability at the intended prediction time is justified.
- The original model-comparison cells choose a best model using test-set results, which makes those comparisons unsuitable as an unbiased final estimate. The cleaned notebook directs users to the separate evaluation script for model selection and holdout evaluation.
- The original workflow applies SMOTE after splitting, but the stronger baseline uses a preprocessing pipeline and class-weighted estimators rather than SMOTE.

The separate baseline script provides a distinct evaluation path with preprocessing fitted within cross-validation folds and model selection confined to the training partition. It has not been evaluated on the project dataset, so no verified performance claims are available from this prepared copy. The notebook and script should be treated as an academic research draft, not as a validated predictive model. Student dropout prediction carries fairness and high-impact decision risks; this work must not be used to make decisions about individual students.

## Attribution and release status

This originated as a team academic project. The original reports contain student IDs and university email addresses and are excluded. Confirm team attribution, dataset permissions, and any course restrictions before publication. No open-source license has been added.
