Live Demo : https://laptop-price-prediction-q7nem7xswse9hzcrlqzb4a.streamlit.app/

# 💻 Laptop Price Prediction

A machine learning project that predicts the price of a laptop (in euros) from its specifications, with a Streamlit web app for live predictions.

**🔗 Live App:** `<add your Streamlit link here>`

---

## 📌 Problem Statement

Laptop prices depend on many things at once: brand, processor, RAM, storage, graphics card, screen size and more. This project builds a regression model that estimates the price of a laptop from its specs, and shows which features drive the price the most.

## 📂 Dataset

- **Source:** Kaggle (laptops dataset)
- **Size:** 2,160 rows and 12 columns (2,159 rows after cleaning)
- **Target:** `Final Price` (in euros)
- **Features:** Status (New/Refurbished), Brand, Model, CPU, RAM, Storage, Storage type, GPU, Screen size, Touch

## 🧹 Data Cleaning

| Problem | What I did |
|---|---|
| `GPU` had 1,371 nulls (63.5%) | Nulls mean no dedicated card, so filled with `Integrated`. I checked this first: laptops with null GPU have an average price of about €942, versus about €1,957 for laptops with a GPU. |
| `Storage type` had 42 nulls | Filled with `SSD` (the dominant type, over 97% of laptops) |
| `Screen` had 4 nulls | Filled with the mean |
| 1 row with `Storage = 0` and no storage info in its title | Dropped |
| Rare brands, CPUs and GPUs | Grouped into `Other` |
| `Laptop` (long title text) and `Model` | Dropped, since their information is already in other columns |
| Skewed target | Applied a log transform (`log1p`) to `Final Price` |

I kept 3 TB and 4 TB storage laptops. They are high-end Asus machines whose prices match their specs, so they are real data and not outliers.

## 📊 EDA Insights

- **Price is right-skewed.** Most laptops cost between €200 and €3,000, with a long tail of premium machines. A log transform was needed for modeling.
- **Eight brands dominate:** Asus, HP, Lenovo, MSI, Acer, Apple, Dell and Microsoft. Most other brands have only 10 to 20 laptops.
- **Razer is the most expensive brand** (average above €3,000), but it has fewer than 50 laptops, so I treat that average with caution.
- **Dedicated GPU roughly doubles the price** compared to integrated graphics.
- **Touchscreen laptops** are more expensive on average.
- **New vs Refurbished:** prices are close, with new laptops slightly higher.
- **Screen size** shows a mild upward trend with price, but its correlation (0.27) is the weakest of the numeric features. RAM and Storage correlate above 0.4 with price.
- **CPU:** Core i9, Ryzen 9 and Apple M2 Pro laptops are the most expensive groups with reliable counts.

## 🤖 Modeling

The target was log-transformed, then predictions were converted back to euros with `expm1`. All models used the same train/test split (80/20, `random_state=42`) for a fair comparison.

| Model | R² | MAE (€) | RMSE (€) |
|---|---|---|---|
| **XGBoost** | **0.825** | **235.8** | **381.2** |
| Random Forest | 0.807 | 241.2 | 400.7 |
| Gradient Boosting | 0.796 | 268.7 | 412.1 |
| Linear Regression | 0.714 | 301.3 | 487.7 |
| Ridge | 0.708 | 300.6 | 492.7 |
| Lasso | 0.269 | 505.2 | 779.7 |

(Lasso scores low because the default `alpha=1` shrinks most features to zero on the log target.)

### 5-Fold Cross-Validation (log-price scale)

| Model | Mean R² | Std | Train R² | Test R² |
|---|---|---|---|---|
| **XGBoost** | **0.839** | 0.016 | 0.927 | 0.846 |
| Random Forest | 0.828 | 0.015 | 0.936 | 0.819 |
| Gradient Boosting | 0.819 | 0.015 | 0.849 | 0.816 |

XGBoost won in every fold. Random Forest overfit the most (train-test gap of 0.117).

### Hyperparameter Tuning (XGBoost, RandomizedSearchCV)

```
subsample: 0.7, reg_lambda: 5, reg_alpha: 0.1, n_estimators: 700,
min_child_weight: 3, max_depth: 5, learning_rate: 0.05, colsample_bytree: 1.0
```

| Metric | Before tuning | After tuning |
|---|---|---|
| CV R² | 0.839 | **0.849** |
| Test R² | 0.846 | **0.851** |
| Train-test gap | 0.081 | **0.062** |
| MAE (€) | 235.8 | **224.2** |
| RMSE (€) | 381.2 | **364.7** |

**Final model:** tuned XGBoost, with an average error of about **€224** per laptop.

## 🔍 What Drives the Price?

Top features from the tuned XGBoost model:

1. **RAM** (by far the most important)
2. **CPU: Intel Celeron** (a strong signal for budget laptops)
3. **Brand: Apple**
4. **GPU: Integrated**
5. **Brand: Razer** and **Brand: Microsoft**
6. Storage, Touch, and high-end GPUs (RTX A2000, GTX 1650, RTX 2060, RTX 3080)

Note: feature importance shows how much the model relies on a feature, not whether it raises or lowers the price.

## ⚠️ Limitations

- The model is accurate for laptops under about €1,500, where most of the data is.
- For premium laptops (€3,000 and above) it often **under-predicts** the price. There are very few such laptops in the data, and tree models cannot extrapolate beyond what they have seen.
- Brands, CPUs and GPUs not seen often in training are treated as `Other`.

## 🚀 Run Locally

```bash
git clone <your-repo-url>
cd <your-repo-folder>

python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

streamlit run apps/app.py
```

## 🗂️ Project Structure

```
├── apps/
│   └── app.py                    # Streamlit app
├── data/
│   ├── laptops.csv               # Raw data
│   └── laptops_cleaned.csv       # Cleaned data
├── model/
│   ├── tuned_xgb_model.pkl       # Trained XGBoost model
│   ├── feature_columns.json      # Training feature columns
│   └── preprocessing.json        # Categories and grouping rules
├── notebook/
│   ├── 01_EDA_Cleaning.ipynb
│   └── 02_Visualization_Model.ipynb
└── requirements.txt
```

## 🛠️ Tech Stack

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Streamlit, Joblib

## 👤 Author

**Munik**
IT Diploma Student, L.E. College, Morbi

- GitHub: `https://github.com/MunikLalluvadiya/laptop-price-prediction.git`
- LinkedIn: `https://www.linkedin.com/in/munik-lalluvadiya/`