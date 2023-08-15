import streamlit as st
import glob
from pathlib import Path
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px

# Add title
st.title("Diary Tone")

filepaths = sorted(glob.glob("diary/*.txt"))
analyzer = SentimentIntensityAnalyzer()


negativity = []
positivity = []
diary_date = []
for filepath in filepaths:
    # Get the date from the filenames
    filename = Path(filepath).stem
    input_date = datetime.strptime(filename, "%Y-%m-%d")
    diary_date.append(input_date.strftime("%b %d"))

    # Sentiment analysis
    with open(filepath, "r") as file:
        content = file.read()
        scores = analyzer.polarity_scores(content)
    negativity.append(scores["neg"])
    positivity.append(scores["pos"])


# Add subheader and figure for positivity tone
st.subheader("Positivity")
figure = px.line(x=diary_date, y=positivity,
                 labels={"x": "Date", "y": "Positivity"})
st.plotly_chart(figure)

# Add subheader and figure for negativity tone
st.subheader("Negativity")
figure = px.line(x=diary_date, y=negativity,
                 labels={"x": "Date", "y": "Negativity"})
st.plotly_chart(figure)