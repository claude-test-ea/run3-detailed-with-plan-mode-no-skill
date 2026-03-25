---
name: Loan ML Project Architecture and Review Findings
description: Architecture decisions, common patterns, and critical bugs found in the loan eligibility prediction ML project
type: project
---

Loan Eligibility Prediction ML project with sklearn pipeline + Streamlit dashboard.

Key architecture: src/ modules (ingestion, preprocessing, training, evaluation, utils), pipeline.py orchestrator, app.py Streamlit entry, streamlit_app/tabs/ for 5 dashboard tabs, tests/ for unit tests.

**Critical bugs fixed 2026-03-25:**
1. Data leakage: preprocessor was fit on entire ml_data before train/val split. Fixed to fit only on training data.
2. Missing `y_val` key in evaluation results dict -- model_deep_dive.py threshold tuning was silently broken.
3. `best_model.pkl` was missing `best_name` and `feature_names` keys that model_deep_dive.py expected.
4. predict_holdout.py column detection didn't match `y_true` column name from predictions CSV.

**Why:** These are the kinds of bugs that pass superficial testing (no exceptions) but produce wrong results or dead features.

**How to apply:** When reviewing future changes, pay special attention to: (a) what keys evaluation.py stores vs what dashboard tabs consume, (b) preprocessor fit/transform ordering relative to train/test splits, (c) CSV column name conventions matching across writer and reader code.
