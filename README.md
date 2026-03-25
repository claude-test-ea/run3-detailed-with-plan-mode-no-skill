# Loan Eligibility Prediction

End-to-end ML pipeline and interactive Streamlit dashboard for predicting loan eligibility.

## Project Structure

```
├── data/input.csv                  # Source dataset (614 rows)
├── src/                            # Core ML modules
│   ├── ingestion.py                # Data loading & splitting
│   ├── preprocessing.py            # Feature engineering & sklearn Pipeline
│   ├── training.py                 # 5 models + GridSearchCV
│   ├── evaluation.py               # Metrics, confusion matrices, ROC curves
│   └── utils.py                    # Shared helpers
├── models/                         # Saved model artifacts
│   ├── model_results.pkl           # All model evaluation results
│   └── best_model.pkl              # Best model (by F1 score)
├── predictions/                    # Prediction CSV files
│   ├── train_predictions.csv
│   ├── validation_predictions.csv
│   └── holdout_predictions.csv
├── streamlit_app/tabs/             # Dashboard tab modules
│   ├── data_explorer.py            # Tab 1: Data Explorer
│   ├── eda_visualizations.py       # Tab 2: EDA & Visualizations
│   ├── model_performance.py        # Tab 3: Model Performance
│   ├── model_deep_dive.py          # Tab 4: Model Deep Dive
│   └── predict_holdout.py          # Tab 5: Hold-Out Predictions
├── tests/                          # Unit tests
├── pipeline.py                     # Standalone ML pipeline script
├── app.py                          # Streamlit dashboard entry point
└── requirements.txt                # Python dependencies
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the ML Pipeline

```bash
python pipeline.py
```

This trains 5 models (Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN), evaluates them, and saves artifacts to `models/` and predictions to `predictions/`.

## Launch the Dashboard

```bash
streamlit run app.py
```

The dashboard has 5 tabs:
1. **Data Explorer** -- Filter and search the full dataset
2. **EDA & Visualizations** -- Interactive Plotly charts
3. **Model Performance** -- Compare all 5 models
4. **Model Deep Dive** -- Feature importances, threshold tuning
5. **Predict on Hold-Out** -- Best model predictions on the last 200 rows

## Run Tests

```bash
python -m pytest tests/ -v
```

## Models Trained

| Model | F1 | Accuracy | ROC AUC |
|---|---|---|---|
| Gradient Boosting | 0.875 | 0.807 | 0.829 |
| SVM | 0.875 | 0.807 | 0.771 |
| Random Forest | 0.864 | 0.795 | 0.788 |
| KNN | 0.836 | 0.759 | 0.759 |
| Logistic Regression | 0.829 | 0.747 | 0.717 |

## Data Split Strategy

- **ML data**: First 414 rows (used for training/validation)
- **Hold-out**: Last 200 rows (never used during training)
- **Train/Val split**: 80/20 stratified within ML data
