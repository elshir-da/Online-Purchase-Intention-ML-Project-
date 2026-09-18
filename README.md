# Online-Purchase-Intention-ML-Project-
A machine learning project analyzing user behavior to predict online purchase conversion.
 Predicting Online Purchase Intention Using Machine Learning

## Project overview
This project uses the UCI Online Shoppers Purchasing Intention dataset to predict whether an e-commerce browsing session results in a purchase (`Revenue`). The goal is to explore how session behaviour can support more targeted, evidence-informed marketing.

**Unit of analysis:** an online shopping session, not an identified individual customer.

## Business problem
E-commerce websites observe many sessions that do not convert. A session-level classifier can help prioritize follow-up experiences and marketing experiments. Predictions should support testing—not be treated as proof that an intervention will increase purchases.

## Dataset and access
- **Dataset:** Online Shoppers Purchasing Intention Dataset
- **Source:** UCI Machine Learning Repository
- **Dataset page:** https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- **Target:** `Revenue` (`True` = purchase; `False` = no purchase)
- The notebook expects the dataset CSV to be available at the path set in its data-loading cell. Download the dataset from UCI and update that path if necessary.

## Workflow
1. Load and inspect the data.
2. Remove 125 exact duplicate rows; the cleaned dataset contains 12,205 rows and 18 columns.
3. Conduct EDA, including target imbalance, visitor types, monthly patterns, weekend/weekday behaviour, and numeric relationships.
4. Perform a stratified 80/20 train/test split (9,764 train; 2,441 test).
5. Engineer 13 session-level features, including total pages/duration, product engagement shares, average time per page, engagement-risk indicators, page-value transformations, seasonality, and visitor-type indicators.
6. Fit preprocessing on training data only. The pipeline expands 30 predictors into 87 model-ready features using one-hot encoding for categorical predictors.
7. Compare a linear SGD baseline, Random Forest baseline, and XGBoost.
8. Tune XGBoost using randomized search and grid search, then explore class-weighting and a probability threshold.

## Models and reported results
The notebook reports these held-out test metrics:

| Model/experiment | Accuracy | Precision (purchase) | Recall (purchase) | F1 (purchase) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| SGDClassifier | 87.51% | 68.00% | 37.00% | 48.00% | 80.83% |
| Random Forest baseline | 89.84% | 70.68% | 59.95% | 64.87% | 91.69% |
| Baseline XGBoost | 89.64% | 69.49% | 60.21% | 64.52% | 91.58% |
| Grid-search XGBoost (default threshold) | 89.76% | 70.25% | 59.95% | 64.69% | 92.73% |
| Final class-weighted XGBoost (weight 4.318742; threshold 0.5) | 86.56% | 54.84% | 80.10% | 65.11% | 92.72% |

Randomized search reported best cross-validation ROC-AUC of 0.9350. Grid search reported best cross-validation ROC-AUC of 0.9367. These are cross-validation scores, not test scores.

The final class-weighted model is **recall-oriented**: it identifies more purchasing sessions, but at the cost of more false positives. Its test confusion matrix is TN=1,807, FP=252, FN=76, TP=306.

A separate threshold exploration using the grid-search model found a threshold of 0.35 with F1=0.6847, precision=0.6443, recall=0.7304, and accuracy=0.8947. This threshold experiment used a different model configuration and should not be presented as the operating threshold of the final class-weighted model.

## Key findings
- The target is imbalanced: approximately 84.4% no-purchase sessions and 15.6% purchase sessions.
- `PageValues` is the dominant feature in the final model’s importance output (importance approximately 0.357). This is model reliance, not evidence of causality.
- Month (especially November), traffic type, visitor type, and product-related engagement also appear among important predictors.
- Probability-based session segments in the notebook:
  - **Low intent:** 1,666 sessions; 40 purchases; 2.40% observed purchase rate.
  - **Medium intent:** 359 sessions; 68 purchases; 18.94% observed purchase rate.
  - **High intent:** 416 sessions; 274 purchases; 65.87% observed purchase rate.
  - Medium + high segments contain 342 of 382 purchases in the test set (89.5%).
- These are session-level segments, not persistent customer profiles.

## Business recommendations (hypotheses to test)
- Test relevant recommendations or streamlined checkout support for high-intent sessions.
- Test modest reminders or incentives for medium-intent sessions.
- Avoid spending equal intervention effort on low-intent sessions without evidence.
- Run controlled A/B tests to estimate incremental impact; model scores alone do not establish uplift.

## Limitations and future work
- `PageValues` may be unavailable early enough in a live session or may encode information close to the purchase outcome. Evaluate a version without it to assess practical usefulness and leakage risk.
- Thresholds should be selected using validation data and explicit costs of false positives and false negatives—not repeatedly optimized on the final test set.
- Assess calibration, temporal/generalization performance, and drift.
- Test interventions with randomized experiments before claiming business impact.
- Consider session-event sequence features if event-level data become available.

## Repository structure
```text
.
├── README.md
├── online_shoppers_intention_project.ipynb
├── online_purchase_intention.py
└── presentation_10min.pptx
```

## Reproducibility
Install the packages used by the notebook/script (for example: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, and `jupyter`). Dataset download and local path configuration may be required. Exact package versions should be recorded in a project environment file before submission.

## Presentation
See [`presentation_10min.pptx`](presentation_10min.pptx). Replace `[Your Name]` and `[GitHub repository URL]` on the title/closing slides before presenting.
