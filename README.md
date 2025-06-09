### 📘 `README.md`

````markdown
# 🏏 IPL Top Players Predictor – Model Development (`IPL.ipynb`)

This notebook (`IPL.ipynb`) outlines the **complete workflow for preparing IPL data**, engineering features, labeling player performance, training a machine learning model, and saving it for deployment. The final model helps predict **top-performing players** for each IPL team before a match.

---

## 🔁 Workflow Summary (Step-by-Step)

### 1. **Import Required Libraries**
The notebook starts by importing:
- `pandas`, `numpy` for data manipulation
- `matplotlib`, `seaborn` for visualization
- `sklearn` modules for model training and evaluation

---

### 2. **Load and Merge Raw Datasets**
- `Deliveries.csv` and `Matches.csv` are loaded.
- Merged on `match_id` to form a unified dataset with detailed match + player performance.

---

### 3. **Initial Data Cleaning**
- Missing values handled (e.g., filling missing city names using venue).
- Duplicate rows removed.
- Data types corrected.

---

### 4. **Player-Level Feature Extraction**
For every match and player, the following statistics are computed:
#### 🏏 Batting:
- `total_runs`, `balls_faced`, `fours`, `sixes`, `strike_rate`

#### 🎯 Bowling:
- `balls_bowled`, `runs_conceded`, `wickets`, `dot_balls`, `economy`

#### 🧤 Fielding:
- `catches`, `run_outs`

- Each player's team is also mapped using match and delivery records.

---

### 5. **Feature Engineering**
Additional features are generated:
- **Recent Form**: average runs and wickets in the last 3 and 5 matches
- **Opponent Stats**: performance vs specific opponent
- **Venue Stats**: performance at specific stadiums
- **Career Totals**: cumulative runs and wickets
- **Averages and Rates**: batting/bowling average, strike rate, economy
- **Player Role Classification** (batsman, bowler, all-rounder, etc.)

---

### 6. **Label Generation**
- A **performance score** is calculated:
  ```python
  label = 1 if (batsman_runs * 1.5 + wickets * 20 + catches * 10 - economy_penalty) is high else 0
````

* Players are labeled as:

  * `1`: top performer
  * `0`: otherwise

---

### 7. **Handle Missing Values and Outliers**

* Venue-based imputation used for missing `city` values.
* Outliers in numerical columns visualized using **boxplots** and handled using IQR or transformation.

---

### 8. **Handle Skewness**

* Features with skewness > ±1 are transformed using:

  * `PowerTransformer` (Yeo-Johnson)
  * `log1p()` for specific columns like `sixes`, `catches`

---

### 9. **Train/Test Split and Model Training**

* `RandomForestClassifier` used to train on player statistics.
* Train-test split performed with stratification.
* Model evaluated using:

  * `Accuracy`, `Precision`, `Recall`, `F1-Score`, `Confusion Matrix`

---

### 10. **Model Evaluation**

* Feature importance plotted using `seaborn`.
* Evaluation scores printed to verify performance (\~90%+ accuracy)

---

### 11. **Save Model**

The trained model is saved as:

```
random_forest_top_players_model.pkl
```



## 🧾 Output Files

* `Player_Stats_With_Features_HandOtl_Sk_Handl_Cleaned.csv` – Final processed data
* `random_forest_top_players_model.pkl` – Trained model for deployment

---

## 📂 Notebook Output Flow

```
├── Load + Merge Data
├── Clean Data
├── Extract Stats per Player
├── Feature Engineering
├── Label Generation
├── Handle Missing & Skewed Data
├── Train Random Forest Classifier
├── Evaluate Model
├── Export Model
```

---




