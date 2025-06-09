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

###  Step 9: Handle Skewness

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

##  Outputs Produced

* `Player_Stats_With_Features_HandOtl_Sk_Handl_Cleaned.csv` – Final dataset with features and labels
* `random_forest_top_players_model.pkl` – Trained ML model to be used in the Streamlit app

---



## **IPL.py – Sequence of Steps in the Code:**

### Step 1: Import Required Libraries
The script starts by importing essential Python libraries:

* `streamlit` for creating the web interface.
* `pandas` for reading and handling the dataset.
* `joblib` for loading the pre-trained machine learning model (`RandomForestClassifier`).

### Step 2: Load the Dataset and the Model
Two functions are defined:

* `load_data()`: loads the cleaned CSV file (`Player_Stats_With_Features_HandOtl_Sk_Handl_Cleaned.csv`) containing player-level features.
* `load_model()`: loads the trained Random Forest model from a `.pkl` file (`random_forest_top_players_model_tuned_O_N.pkl`).
  Both are decorated with Streamlit caching to improve performance during reruns.

### Step 3: Configure the Streamlit App
The layout of the Streamlit page is configured, and the app title is displayed: “Top Players Predictor Before Match based on 2008 to 2019 IPL Analysis”.

### Step 4: Team Selection
The user is presented with a dropdown menu containing all unique teams found in the dataset. Once the user selects a team, the dataset is filtered to only include rows corresponding to that team.

### Step 5: Prepare Data for Prediction
Unnecessary columns are dropped from the filtered dataset. These include columns that are not useful for prediction such as `match_id`, `date`, `venue`, and others. Only feature columns relevant for prediction are retained.

### Step 6: Predict Top Performers
The filtered feature data is passed to the loaded machine learning model. The model returns a binary prediction (`1` for top performer, `0` otherwise), which is appended as a new column in the team data.

### Step 7: Filter Top Players and Remove Redundancy
From the prediction results, only the rows with prediction value `1` are selected. The player entries are grouped to remove duplicates using groupby (if implemented), and average stats are aggregated across matches for each player.

### Step 8: Sort and Select Top 11 Players
The top performers are sorted based on multiple performance metrics:

* High `total_runs`
* High `wickets`
* High `strike_rate`
* Low `economy`
  Then, the top 11 players are selected. If fewer than 11 unique top performers exist, it displays as many as are available without duplication.

### Step 9: Display Results in Streamlit
The final top players list is displayed in a styled table within the Streamlit app. The user can visually review the stats of predicted top performers.

### Step 10: Enable CSV Download
A download button is provided, allowing the user to export the selected top player list as a `.csv` file named after the selected team.
