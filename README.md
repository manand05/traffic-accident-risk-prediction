# Traffic Accident Risk Prediction

ML system for predicting traffic accident risk zones to optimize emergency response times across Houston and Miami.

---

## Project Overview

Emergency response systems today deploy ambulances reactively, after a call is received. This project predicts **where and when** accidents are most likely to occur, enabling proactive ambulance repositioning to reduce response times.

The system uses a **Hurdle Model** architecture:

- **Step 1 (Classifier):** Predicts whether an accident will occur in a given zone/hour
- **Step 2 (Regressor):** Given an accident is predicted, estimates expected severity
- **Final Risk Score:** `P(Accident) × E[Count | Accident]`

---

## Folder Structure

```
TRAFFIC-ACCIDENT-RISK-PREDICTION/
│
├── dataset_modelling/               # All modelling notebooks
│   ├── features.py                  # Shared feature engineering function (imported by all notebooks)
│   ├── Logistic_Regression.ipynb    # LR classifier + XGB regressor hurdle model
│   ├── Poisson_regression.ipynb     # Poisson regression (baseline attempt, shows why it fails)
│   ├── RF_POC.ipynb                 # Random Forest proof of concept
│   ├── RF_tuned.ipynb               # Tuned Random Forest classifier
│   ├── XGBoost_POC.ipynb            # XGBoost proof of concept
│   └── XGBoost_tuned.ipynb          # Final tuned XGBoost hurdle model (champion)
│
├── datasets/
│   ├── modelling_dataset.parquet    # Cleaned and preprocessed dataset (4.7M rows)
│   ├── final_results.parquet        # Test set predictions with risk scores
│   ├── test_df_eng.parquet          # Engineered test set
│   ├── X_test_classifier.parquet    # Test features for classifier evaluation
│   └── y_test_classifier.csv        # Test labels for classifier evaluation
│
├── models/
│   ├── xgb_classifier_model.pkl     # XGBoost classifier (POC)
│   ├── xgb_classifier_model2.pkl    # XGBoost classifier (tuned)
│   ├── xgb_regressor_model.pkl      # XGBoost regressor (POC)
│   ├── xgb_regressor_model2.pkl     # XGBoost regressor (tuned, used across all comparisons)
│   └── lr_classifier_model.pkl      # Logistic Regression classifier + scaler bundle
│
├── dataset_modelling.ipynb          # EDA and initial exploration
├── dataset_preprocessing.ipynb      # Full preprocessing pipeline
├── model_evaluation.ipynb           # Cross-model comparison and final evaluation
├── README.md
└── requirements.txt
```

---

## How to Run the Code

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run preprocessing (only needed once)

```bash
# Open and run all cells in:
dataset_preprocessing.ipynb
# Output: datasets/modelling_dataset.parquet
```

### 3. Run modelling notebooks (in any order)

```bash
# Champion model:
dataset_modelling/XGBoost_tuned.ipynb

# Comparison models:
dataset_modelling/Logistic_Regression.ipynb
dataset_modelling/RF_tuned.ipynb

# Baseline / diagnostic:
dataset_modelling/Poisson_regression.ipynb
```

### 4. Compare all models

```bash
model_evaluation.ipynb
```

> **Note:** All notebooks load data from `../datasets/modelling_dataset.parquet` and save models to `../models/`. Run from within the `dataset_modelling/` directory or adjust paths accordingly.

---

## Dependencies

```
pandas
numpy
kagglehub
scikit-learn
matplotlib
seaborn
jupyter
ipykernel
h3
plotly
statsmodels
patsy
xgboost
pyarrow        # for parquet files
holidays       # for holiday feature generation
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## Reproducibility Instructions

All randomness is controlled via `random_state=42` throughout. To fully reproduce results:

1. Use Python 3.9+
2. Install exact versions from `requirements.txt`
3. Run `dataset_preprocessing.ipynb` first to regenerate `modelling_dataset.parquet`
4. Run modelling notebooks top-to-bottom without skipping cells
5. **Do not use `train_test_split`** all splits are chronological to prevent data leakage:
   - Train: before 2021-03-16
   - Validation: 2021-03-16 to 2022-03-23
   - Test: after 2022-03-23

---

## Dataset Description

**Source:** [US Accidents (2016–2023) — Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents/data)

**Scope:** Filtered to Houston and Miami for computational feasibility.

**Target variable:** `Accident_Count`: number of accidents in a ~ 4km × 4km grid zone per hour. Binarized to `Is_Accident` (0/1) for classification.

### Preprocessed Feature Set

| Column                      | Type     | Description                                   |
| :-------------------------- | :------- | :-------------------------------------------- |
| `Time_Stamp`                | datetime | Hourly timestamp                              |
| `Year`                      | int      | Year extracted from timestamp                 |
| `Hour`                      | int      | Hour of day (0–23)                            |
| `Day_of_Week`               | int      | Day of week (0=Monday)                        |
| `Month`                     | int      | Month (1–12)                                  |
| `Weekend`                   | int      | 1 if Saturday or Sunday                       |
| `Holiday`                   | int      | 1 if US public holiday                        |
| `Zone_Int_ID`               | int      | Integer ID for ~ 4km × 4km grid zone          |
| `Amenity`                   | float    | Proportion of accidents near an amenity POI   |
| `Crossing`                  | float    | Proportion near a road crossing               |
| `Give_Way`                  | float    | Proportion near a give way sign               |
| `Junction`                  | float    | Proportion near a junction                    |
| `Railway`                   | float    | Proportion near a railway                     |
| `Station`                   | float    | Proportion near a station                     |
| `Stop`                      | float    | Proportion near a stop sign                   |
| `Traffic_Signal`            | float    | Proportion near a traffic signal              |
| `City_Houston`              | bool     | 1 if zone is in Houston                       |
| `City_Miami`                | bool     | 1 if zone is in Miami                         |
| `Temperature(F)`            | float    | Temperature in Fahrenheit                     |
| `Humidity(%)`               | float    | Humidity percentage                           |
| `Pressure(in)`              | float    | Air pressure in inches                        |
| `Visibility(mi)`            | float    | Visibility in miles                           |
| `Wind_Speed(mph)`           | float    | Wind speed in mph                             |
| `Precipitation(in)`         | float    | Precipitation in inches                       |
| `Weather_Clear`             | float    | 1 if weather is clear                         |
| `Weather_Cloudy`            | float    | 1 if weather is cloudy                        |
| `Weather_Dust/Windy`        | float    | 1 if dusty or windy                           |
| `Weather_Rain/Drizzle`      | float    | 1 if rain or drizzle                          |
| `Weather_Snow/Ice`          | float    | 1 if snow or ice                              |
| `Weather_Stormy`            | float    | 1 if stormy conditions                        |
| `Weather_Visibility Issues` | float    | 1 if visibility is impaired                   |
| `Accident_Count`            | int      | Number of accidents in zone per hour (target) |

---

## Feature Engineering

All engineered features live in `dataset_modelling/features.py` and are imported by every modelling notebook to ensure consistency. They are computed from training data only — zone statistics are fit on training rows and applied to validation and test to prevent data leakage.

| Feature                           | Description                                                                                     |
| :-------------------------------- | :---------------------------------------------------------------------------------------------- |
| `Wind_x_Precip`                   | Wind speed × precipitation captures combined hazardous weather effect                           |
| `Hour_x_Weekend`                  | Hour × weekend flag captures weekend rush hour differences                                      |
| `BadWeather`                      | 1 if any of rain, stormy, or visibility issues present (clipped 0–1)                            |
| `BadWeather_x_Hour`               | Bad weather × hour captures when bad weather coincides with busy periods                        |
| `Hour_squared`                    | Non-linear transformation of hour captures U-shaped accident patterns across the day            |
| `WindSpeed_squared`               | Non-linear wind speed high winds disproportionately increase risk                               |
| `Precip_squared`                  | Non-linear precipitation heavy rain is far more dangerous than light rain                       |
| `Hour_sin` / `Hour_cos`           | Cyclic encoding of hour ensures model knows hour 23 and hour 0 are adjacent, not 23 units apart |
| `Month_sin` / `Month_cos`         | Cyclic encoding of month December and January treated as adjacent                               |
| `DayOfWeek_sin` / `DayOfWeek_cos` | Cyclic encoding of day Sunday and Monday treated as adjacent                                    |
| `Zone_Mean`                       | Historical mean accident count for this zone (from training data only)                          |
| `Zone_Std`                        | Historical standard deviation of accident count for this zone                                   |
| `Zone_Max`                        | Historical maximum accident count ever recorded in this zone                                    |

---
