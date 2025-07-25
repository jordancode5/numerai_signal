## Warning

In my own testing, there is a sharp turn around 2016 where using logistic regression and even gradient boosted logistic regression has a sharp alpha decay (without very creative periods and features). It appears that the introduction and adoption of TensorFlow (in 2016) libraries may have caused a dramatic shift in market dynamics.

## 🧠 Logistic Regression Baselines

This section contains two logistic regression-style models trained on the Numerai dataset: one using `scikit-learn` and another using `LightGBM`. These models serve as quick, interpretable baselines to understand the predictive structure of the data and establish initial benchmark metrics.

---

### 🔹 `scikit-learn` Logistic Regression

A classic binary classifier that maps feature vectors to probabilities using a sigmoid function.

#### 🚀 What This Does

- Loads preprocessed Numerai data  
- Selects one or more classification targets  
- Trains a `LogisticRegression` model from `scikit-learn`  
- Evaluates performance using:
  - Area Under the Curve (AUC)
  - Accuracy

### 🔹 `lightGBM` Logistic Regression

Logistic Regression Plus

#### 🚀 What This Does

- Does logistic regression, but better
- Takes into account potential nonlinear relationships and interactions
- Not too out of control for feature prediction (yet) 

#### 📈 Sample Output

Training model for target_raw_return_20...
Accuracy: 0.7541
AUC Score: 0.5987

Training model for target_factor_neutral_20...
Accuracy: 0.7413
AUC Score: 0.5821

#### ✅ Why Use This?

- Fast and interpretable
- Acts as a benchmark against more complex models
- Helps check for potential signal or data leakage early in the workflow

---

### 🔸 LightGBM Logistic Regression (`LGBMClassifier`)

LightGBM’s `LGBMClassifier` offers gradient-boosted decision trees with logistic loss for classification.

#### 🔧 Configuration

```python
model = LGBMClassifier(
    n_estimators=100,
    objective="binary",
    verbosity=-1,
    random_state=42
)
```

---

#### 🔍 A file summary

- Download_data.ipynb: Simply gives an introduction of how to download data and see what you are working with
- Quant_mk2_ipynb: Gives you a simple exercise of pulling, processing, splitting, training, and testing data with included histograms to visualize progress
- Quant_mk3_ipynb: Markets are not a static thing; they shift with time. This is where you start making decisions on implementing a dynamic rather than static modeling framework. Using a walk-forward approach, you can adapt to current market regimes while maintaining some of the context of your past data. Be aware, there are MAJOR MAJOR MAJOR (do you get it?) tradeoffs between the amount and type of data you pick, the adaptability of your model, and the compute requirements of your model. 
- 
