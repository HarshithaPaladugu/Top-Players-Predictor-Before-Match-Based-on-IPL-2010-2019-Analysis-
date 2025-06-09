# IPL Top Players Prediction – Model Development (`IPL.ipynb`)

This Jupyter notebook implements a full machine learning pipeline to predict **top-performing IPL players** from the seasons **2008 to 2019**. The final goal is to identify 11 key players for any team before a match, based on their statistical performance using a Random Forest classifier.

---

## Notebook Workflow – Step-by-Step

### Step 1: Import Required Libraries

- Imports essential libraries such as `pandas`, `numpy`, `seaborn`, `matplotlib`, and `sklearn`.
- Ensures all visualization and ML tools are available.

---

###  Step 2: Load & Merge Raw Data

- Loads `Deliveries.csv` and `Matches.csv`.
- Merges the two datasets on `match_id` to form a comprehensive data frame with ball-by-ball and match-level information.

---

###  Step 3: Preprocess Data

- Filters only relevant seasons (e.g., 2008–2019).
- Cleans missing values (e.g., missing cities imputed from venue).
- Ensures proper data types and removes unwanted records.

---

###  Step 4: Create Player-Level Match Stats

For each player per match, the following are calculated:

#### Batting Metrics:
- `total_runs`, `balls_faced`, `fours`, `sixes`, `strike_rate`

#### Bowling Metrics:
- `balls_bowled`, `runs_conceded`, `wickets`, `dot_balls`, `economy`

#### Fielding Metrics:
- `catches`, `run_outs`

Also maps each player to their respective team in that match.

---

###  Step 5: Feature Engineering

Adds the following **aggregated features**:

- **Recent Form**: performance in last 3 and 5 matches
- **Opponent-wise Averages**: runs/wickets vs opponent
- **Venue-wise Averages**: performance at particular venues
- **Career Stats**: overall runs/wickets across seasons
- **Derived Stats**:
  - Batting Average
  - Bowling Average
  - Economy
  - Player Role (batsman, bowler, all-rounder, etc.)

---

### Step 6: Label Generation

- A custom **performance score** is computed using:
  ```python
  label = 1 if (batsman_runs * 1.5 + wickets * 20 + catches * 10) - economy_penalty > threshold else 0

* A new `label` column is created where:

  * `1` = Top performer
  * `0` = Not a top performer

---

###  Step 7: Handle Missing Values

* Handles missing team/city data.
* Uses mapping strategies (e.g., filling city based on venue mode).
* Removes or fills `NaN` values in numerical fields.

---

### Step 8: Outlier Detection & Removal

* Uses box plots to visualize and detect outliers for:

  * `total_runs`, `balls_faced`, `sixes`, `wickets`, `catches`, etc.
* Optionally applies IQR filtering or manual inspection.

---

### ✅ Step 9: Handle Skewness

* Uses `PowerTransformer` and `np.log1p()` to normalize:

  * `sixes`, `wickets`, `catches`, etc.
* Recalculates skewness and confirms improvement.

---

### Step 10: Train-Test Split

* Splits data into `X` (features) and `y` (labels).
* Uses `train_test_split()` with `stratify=y`.

---

###  Step 11: Train Random Forest Classifier

* Fits a `RandomForestClassifier` with default or tuned hyperparameters.
* Evaluates performance using:

  * Accuracy
  * Precision
  * Recall
  * F1-Score
  * Classification report

---

### Step 12: Feature Importance

* Uses `.feature_importances_` from the trained model to rank features.
* Plots a bar chart showing which features contribute most.

---

### Step 13: Save the Trained Model

* Saves the model to a `.pkl` file using `joblib`:

  ```python
  joblib.dump(model, "random_forest_top_players_model.pkl")
  ```

---

## 📁 Outputs Produced

* `Player_Stats_With_Features_HandOtl_Sk_Handl_Cleaned.csv` – Final dataset with features and labels
* `random_forest_top_players_model.pkl` – Trained ML model to be used in the Streamlit app

---


