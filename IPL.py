import streamlit as st
import pandas as pd
import joblib

# Load the model and dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Player_Stats_With_Features_HandOtl_Sk_Handl_Cleaned.csv")
    return df

@st.cache_resource
def load_model():
    return joblib.load("random_forest_top_players_model.pkl")

df = load_data()
model = load_model()

# Set up the layout
st.set_page_config(page_title="Top 11 Player Predictor", layout="wide")
st.title("🏏 Top 11 Players Predictor Before Match")

# Team selection
teams = sorted(df['team'].unique())
selected_team = st.selectbox("Select a Team", teams)

# Filter data for selected team
team_df = df[df['team'] == selected_team].copy()

# Drop unnecessary columns for prediction
drop_cols = ['match_id', 'player', 'team', 'season', 'date', 'venue', 'city',
             'batting_team', 'bowling_team', 'team1', 'team2', 'label']
X_team = team_df.drop(columns=drop_cols, errors='ignore')

# Predict
predictions = model.predict(X_team)
team_df['Prediction'] = predictions

# Filter top 11 players
top_11 = team_df[team_df['Prediction'] == 1].copy()
top_11 = top_11.sort_values(by=['total_runs', 'wickets', 'strike_rate', 'economy'], ascending=[False, False, False, True]).head(11)

# Display
st.subheader(f"Top 11 Predicted Players for {selected_team}")
if not top_11.empty:
    st.dataframe(top_11[['player', 'total_runs', 'sixes', 'strike_rate',
                         'balls_bowled', 'runs_conceded', 'wickets',
                         'dot_balls', 'economy', 'catches', 'run_outs']])
else:
    st.warning("No top players predicted for the selected team.")

# Optional: Download CSV
st.download_button("Download Top 11 as CSV", top_11.to_csv(index=False), file_name=f"{selected_team}_top_11.csv")
