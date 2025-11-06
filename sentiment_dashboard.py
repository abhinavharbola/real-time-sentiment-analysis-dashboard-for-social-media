import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, select, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
import datetime
import time
import re
from collections import Counter
# --- New Imports ---
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- Database Setup (Mirrors backend_api.py) ---
# This allows Streamlit to read from the same DB
DATABASE_URL = "sqlite:///./sentiment.db"
# `check_same_thread=False` is required for SQLite in this multi-threaded context
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SQLAlchemy Model (must be defined again here)
class Tweet(Base):
    __tablename__ = "tweets"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String)
    sentiment_label = Column(String)
    sentiment_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

# --- Utility Functions (Mirrors backend_api.py) ---

def clean_text(text: str) -> str:
    """Removes URLs, mentions, and hashtags for cleaner word clouds."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    text = re.sub(r'#\w+', '', text)     # Remove hashtags
    text = re.sub(r'\W', ' ', text)     # Remove non-alphanumeric
    text = text.lower()
    return text

def get_word_frequencies(texts: list[str]) -> dict:
    """
    Calculates word frequencies for word cloud.
    Returns a dictionary: {'word': count}
    """
    # Standard English stopwords
    stopwords = set([
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
        "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", 
        "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", 
        "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", 
        "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", 
        "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", 
        "about", "against", "between", "into", "through", "during", "before", "after", "above", 
        "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", 
        "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", 
        "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", 
        "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
    ])
    
    all_words = " ".join(texts).split()
    # Filter out stopwords and short words
    filtered_words = [word for word in all_words if word not in stopwords and len(word) > 2]
    word_counts = Counter(filtered_words)
    # Return a dictionary, which is what the 'wordcloud' library prefers
    return dict(word_counts)

# --- New Helper Function for Plotting ---

def generate_wordcloud_image(word_freq_dict: dict, positive: bool = True):
    """Generates a Matplotlib figure of a word cloud."""
    
    # Set colors based on sentiment
    if positive:
        color_func = lambda *args, **kwargs: "green"
    else:
        color_func = lambda *args, **kwargs: "red"

    # Check if frequency dictionary is empty
    if not word_freq_dict:
        # Create an empty figure as a placeholder
        fig, ax = plt.subplots(figsize=(8, 4)) # Match aspect ratio
        ax.text(0.5, 0.5, "No data to display", 
                horizontalalignment='center', 
                verticalalignment='center', 
                transform=ax.transAxes,
                fontsize=12, color='gray')
        ax.axis("off")
        return fig

    # Initialize the WordCloud object
    wc = WordCloud(
        background_color="white",
        max_words=50,
        width=800,
        height=400,
        color_func=color_func, # Apply our color function
        prefer_horizontal=1.0 # All words horizontal
    )
    
    # Generate the cloud from frequencies
    try:
        wc.generate_from_frequencies(word_freq_dict)
    except ValueError:
        # Handle rare case where all words are filtered out
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data to display", 
                horizontalalignment='center', 
                verticalalignment='center', 
                transform=ax.transAxes,
                fontsize=12, color='gray')
        ax.axis("off")
        return fig
    
    # Create a matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off") # Hide the axes
    
    return fig

# --- Data Loading Function ---

def load_dashboard_data(db: Session):
    """
    Queries the database and aggregates data for the dashboard.
    This is the core logic for the Streamlit app.
    """
    
    # Define the time window (e.g., last 10 minutes)
    time_window = datetime.datetime.utcnow() - datetime.timedelta(minutes=10)
    
    # 1. Get KPIs (Total)
    total_counts_query = (
        select(Tweet.sentiment_label, func.count(Tweet.sentiment_label))
        .group_by(Tweet.sentiment_label)
    )
    total_counts_result = db.execute(total_counts_query).all()
    total_counts = {label: count for label, count in total_counts_result}

    # 2. Get KPIs (Window)
    window_counts_query = (
        select(Tweet.sentiment_label, func.count(Tweet.sentiment_label))
        .where(Tweet.timestamp >= time_window)
        .group_by(Tweet.sentiment_label)
    )
    window_counts_result = db.execute(window_counts_query).all()
    window_counts = {label: count for label, count in window_counts_result}

    kpis = {
        "total_positive": total_counts.get("POSITIVE", 0),
        "total_negative": total_counts.get("NEGATIVE", 0),
        "window_positive": window_counts.get("POSITIVE", 0),
        "window_negative": window_counts.get("NEGATIVE", 0),
        "total_processed": sum(total_counts.values())
    }
    
    # 3. Get Time-Series Data (per-minute for last 10 mins)
    time_series_data = []
    # Loop from 10 minutes ago (i=10) to now (i=0)
    for i in range(10, -1, -1):
        start_time = datetime.datetime.utcnow() - datetime.timedelta(minutes=i+1)
        end_time = datetime.datetime.utcnow() - datetime.timedelta(minutes=i)
        
        minute_query = (
            select(Tweet.sentiment_label, func.count(Tweet.sentiment_label))
            .where(Tweet.timestamp >= start_time)
            .where(Tweet.timestamp < end_time)
            .group_by(Tweet.sentiment_label)
        )
        minute_result = db.execute(minute_query).all()
        minute_counts = {label: count for label, count in minute_result}
        
        time_series_data.append({
            "Time": f"{i}m ago", # Label for the x-axis
            "Sentiment": "POSITIVE",
            "Count": minute_counts.get("POSITIVE", 0),
        })
        time_series_data.append({
            "Time": f"{i}m ago", # Label for the x-axis
            "Sentiment": "NEGATIVE",
            "Count": minute_counts.get("NEGATIVE", 0),
        })
    
    time_series_df = pd.DataFrame(time_series_data)
    
    # 4. Get Recent Tweets
    recent_tweets_query = (
        select(Tweet)
        .where(Tweet.timestamp >= time_window)
        .order_by(Tweet.timestamp.desc())
        .limit(10)
    )
    recent_tweets = db.execute(recent_tweets_query).scalars().all()
    recent_tweets_df = pd.DataFrame(
        [{
            "Time": t.timestamp.strftime("%H:%M:%S"), 
            "Sentiment": t.sentiment_label, 
            "Tweet": t.text
        } for t in recent_tweets]
    )

    # 5. Get Word Cloud Data (from last 10 mins, up to 100 tweets for performance)
    positive_texts = [
        clean_text(row[0]) for row in db.execute(
            select(Tweet.text)
            .where(Tweet.timestamp >= time_window)
            .where(Tweet.sentiment_label == "POSITIVE")
            .limit(100)
        ).all()
    ]
    positive_word_cloud_freq = get_word_frequencies(positive_texts)
    
    negative_texts = [
         clean_text(row[0]) for row in db.execute(
            select(Tweet.text)
            .where(Tweet.timestamp >= time_window)
            .where(Tweet.sentiment_label == "NEGATIVE")
            .limit(100)
        ).all()
    ]
    negative_word_cloud_freq = get_word_frequencies(negative_texts)

    return kpis, time_series_df, recent_tweets_df, positive_word_cloud_freq, negative_word_cloud_freq

# --- Streamlit App Layout ---

# Set page config to wide mode for better dashboard layout
st.set_page_config(layout="wide")

st.title("Real-Time Sentiment Dashboard 📈")

# Create a single placeholder for the entire dashboard
# This allows us to overwrite the whole page on refresh
placeholder = st.empty()

# Auto-refresh loop
while True:
    try:
        # Create a new DB session for this iteration
        db = SessionLocal()
        
        # Load all data
        kpis, time_series_df, recent_tweets_df, pos_word_freq, neg_word_freq = load_dashboard_data(db)
        
        # Close the session
        db.close()
        
        # Use the placeholder to draw the dashboard
        with placeholder.container():
            # --- Row 1: KPIs ---
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            
            kpi_col1.metric(
                label="Total Tweets Processed",
                value=f"{kpis['total_processed']:,}"
            )
            kpi_col2.metric(
                label="Sentiment (Last 10 Mins)",
                value=f"{kpis['window_positive']} Pos / {kpis['window_negative']} Neg"
            )
            kpi_col3.metric(
                label="All-Time Positive",
                value=f"{kpis['total_positive']:,}",
                delta_color="normal" # Green
            )
            kpi_col4.metric(
                label="All-Time Negative",
                value=f"{kpis['total_negative']:,}",
                delta_color="inverse" # Red
            )
            
            st.divider()

            # --- Row 2: Charts ---
            chart_col1, chart_col2 = st.columns([2, 1]) # 2/3 and 1/3 layout
            
            with chart_col1:
                st.subheader("Sentiment Over Time (Last 10 Mins)")
                if not time_series_df.empty:
                    # Use Streamlit's native line chart
                    st.line_chart(
                        time_series_df,
                        x="Time",
                        y="Count",
                        color="Sentiment" # Creates two lines (Positive/Negative)
                    )
                else:
                    st.warning("No time-series data yet.")
                
                st.subheader("Live Tweet Feed (Last 10 Mins)")
                # Display recent tweets in a table
                st.dataframe(
                    recent_tweets_df,
                    width='stretch', # Replaced use_container_width=True
                    hide_index=True
                )

            with chart_col2:
                st.subheader("Overall Sentiment")
                # Create a simple DataFrame for the bar chart
                total_sentiment_df = pd.DataFrame({
                    "Sentiment": ["POSITIVE", "NEGATIVE"],
                    "Count": [kpis['total_positive'], kpis['total_negative']]
                })
                st.bar_chart(
                    total_sentiment_df,
                    x="Sentiment",
                    y="Count",
                    color="Sentiment" # Colors bars based on sentiment
                )

                st.subheader("Recent Positive Keywords")
                # Generate and display the word cloud image
                pos_fig = generate_wordcloud_image(pos_word_freq, positive=True)
                st.pyplot(pos_fig, width='stretch') # Replaced use_container_width=True


                st.subheader("Recent Negative Keywords")
                # Generate and display the word cloud image
                neg_fig = generate_wordcloud_image(neg_word_freq, positive=False)
                st.pyplot(neg_fig, width='stretch') # Replaced use_container_width=True


    except Exception as e:
        # If the DB is empty or locked, just show a warning and continue
        with placeholder.container():
            st.warning(f"Waiting for data... Is the backend running? (Error: {e})")
    
    # Wait for 3 seconds before refreshing
    time.sleep(3)