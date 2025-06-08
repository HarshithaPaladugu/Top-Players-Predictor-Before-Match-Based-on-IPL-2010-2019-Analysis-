import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# Load Dataset and Model
# ---------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("G:\\CAP_Guvi\\Player_Stats_With_Features_HandOtl_Sk_Handl_Cleaned.csv")
        return df
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_model():
    try:
        model = joblib.load("G:\\CAP_Guvi\\random_forest_top_players_model_tuned_O_N.pkl")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

# ---------------------------
# Main App Code
# ---------------------------
def main():
    st.set_page_config(page_title="Top Player Predictor", layout="wide")
    st.title("🏏 Top Players Predictor Before Match (Based on IPL 2010–2019 Analysis)")

    df = load_data()
    model = load_model()

    if df.empty or model is None:
        st.error("❗ Exiting: Data or model not available.")
        return

    # ---------------------------
    # Team Selection
    # ---------------------------
    teams = sorted(df['team'].dropna().unique())
    selected_team = st.selectbox("Select a Team", teams)

    team_df = df[df['team'] == selected_team].copy()
    total_unique_players = team_df['player'].nunique()
    st.write(f"ℹ️ Total unique players found for {selected_team}: {total_unique_players}")

    if team_df.empty:
        st.warning("No data found for the selected team.")
        return

    # Drop unnecessary columns before prediction
    drop_cols = ['match_id', 'team', 'season', 'date', 'venue', 'city',
                 'batting_team', 'bowling_team', 'team1', 'team2', 'label']
    X_team = team_df.drop(columns=drop_cols, errors='ignore')

    if 'player' in X_team.columns:
        player_names = X_team['player']
        X_features = X_team.drop(columns=['player'])
    else:
        player_names = team_df['player']
        X_features = X_team

    if X_features.empty:
        st.warning("⚠️ No usable feature columns available for the selected team.")
        return

    # ---------------------------
    # Make Predictions
    # ---------------------------
    try:
        predictions = model.predict(X_features)
        team_df['Prediction'] = predictions
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        return

    # ---------------------------
    # Filter & Aggregate Top Players
    # ---------------------------
    top_df = team_df[team_df['Prediction'] == 1].copy()

    if top_df.empty:
        st.warning("⚠️ No players predicted as top performers.")
        return

    top_unique_players = top_df['player'].nunique()
    st.write(f"✅ Unique players predicted as top: {top_unique_players}")

    # Aggregate stats per unique player
    agg_cols = ['total_runs', 'sixes', 'strike_rate', 'balls_bowled', 'runs_conceded',
                'wickets', 'dot_balls', 'economy', 'catches', 'run_outs']
    top_summary = top_df.groupby('player')[agg_cols].mean().reset_index()

    # Sort and select up to top 11
    top_11 = top_summary.sort_values(
        by=['total_runs', 'wickets', 'strike_rate', 'economy'],
        ascending=[False, False, False, True]
    ).head(11)

    num_top_players = len(top_11)

    # Reset index for display and export
    top_11 = top_11.reset_index(drop=True)
    csv_data = top_11.to_csv(index=False)

    # ---------------------------
    # Display Results
    # ---------------------------
    st.subheader(f"🏆 Top {num_top_players} Player{'s' if num_top_players > 1 else ''} for {selected_team}")

    if not top_11.empty:
        st.dataframe(top_11)

        # Download button
        st.download_button(
            label=f"Download Top {num_top_players} Players as CSV",
            data=csv_data,
            file_name=f"{selected_team}_top_{num_top_players}.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ No top players to display.")

# 🚀 Run the app
if __name__ == "__main__":
    main()
