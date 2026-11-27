# Startup Operating Status Prediction

A machine learning project evaluating **Logistic Regression** and **Decision Tree Classifier** models to predict whether a startup remains actively `operating` (`status == 2`) based on historical funding rounds, financial metrics, and company metadata.

---

## 📊 Dataset Overview

* **Source File:** `cleaned_data.csv.xls`
* **Total Records:** 54,294 startups (`status` breakdown: `47,999` operating, `3,692` acquired, `2,603` closed)
* **Features Included:**
  * **Identifiers & Metadata:** `permalink`, `name`, `homepage_url`, `category_list`, `market`, `country_code`, `state_code`, `region`, `founded_year`, and one-hot encoded `founded_quarter` columns (expanding the feature space to 4,863 columns)
  * **Funding Metrics:** `funding_total_usd`, `funding_rounds`, `seed`, `venture`, `equity_crowdfunding`, `undisclosed`, `convertible_note`, `debt_financing`, `angel`, `grant`, `private_equity`, `post_ipo_equity`, `post_ipo_debt`, `secondary_market`, `product_crowdfunding`, and rounds `round_A` through `round_H`

---

## ⚙️ Methodology & Workflow

1. **Data Cleaning & Preprocessing**
   * Verified zero missing values (`isnull().sum()`) across the 4,863 preprocessed features.
   * Cleaned `funding_total_usd` by stripping commas, replacing `-` placeholders with `0`, and converting values to `float`.
2. **Target Engineering**
   * Created a binary classification variable (`target`):
     * `1` if `status == 2` (Operating)
     * `0` otherwise (`status` in `{0, 1}`)
3. **Feature Selection & Train-Test Split**
   * Dropped non-numeric/high-cardinality text identifiers (`permalink`, `name`, `homepage_url`, `category_list`, `state_code`, `region`) and `target` to construct the feature matrix `X`.
   * Split the dataset into **80% Training** and **20% Testing** sets (`random_state=2`).
4. **Model Training & Evaluation**
   * **Logistic Regression:** Trained with `solver="liblinear"` and `max_iter=200`.
   * **Decision Tree Classifier:** Trained using default `DecisionTreeClassifier` hyperparameters on the same feature split.

---

## 📈 Model Performance

| Model | Hyperparameters / Notes | Test Accuracy |
| :--- | :--- | :---: |
| **Logistic Regression** | `solver="liblinear"`, `max_iter=200` | `88.35%` (`0.8835`) |
| **Decision Tree Classifier** | Default parameters | `100.00%` (`1.0`) |

---

## 🛠️ Tech Stack

* **Language:** Python 3
* **Environment:** Google Colab / Jupyter Notebook
* **Libraries:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`

---

## 🚀 Getting Started

### Prerequisites
Install the required dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn jupyter
