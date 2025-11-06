import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, select, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from transformers import pipeline
import datetime
import re
from collections import Counter
import logging

# --- Setup Logging ---
# Set logging level for transformers to reduce spam
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- Database Setup ---
DATABASE_URL = "sqlite:///./sentiment.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SQLAlchemy Model
class Tweet(Base):
    __tablename__ = "tweets"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String)
    sentiment_label = Column(String)
    sentiment_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

# Pydantic Model (for request body)
class TweetInput(BaseModel):
    text: str

# --- ML Model Loading ---
# Load the sentiment analysis pipeline from Hugging Face
# Using a distilled version for better performance
try:
    print("Loading sentiment analysis model...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    sentiment_pipeline = None

# --- FastAPI App Initialization ---
app = FastAPI(title="Sentiment Analysis API")

# CORS (Cross-Origin Resource Sharing) Middleware
# This allows our React frontend (running on localhost:3000)
# to talk to our backend (running on localhost:8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Utility Functions ---

def clean_text(text: str) -> str:
    """Removes URLs, mentions, and hashtags for cleaner word clouds."""
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    text = re.sub(r'#\w+', '', text)     # Remove hashtags
    text = re.sub(r'\W', ' ', text)     # Remove non-alphanumeric
    text = text.lower()
    return text

def get_word_frequencies(texts: list[str]) -> list[dict]:
    """Calculates word frequencies for word cloud."""
    # A basic list of stopwords
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
    
    # Format for react-wordcloud: { text: "word", value: 64 }
    return [{"text": word, "value": count} for word, count in word_counts.most_common(50)]

# --- API Endpoints ---

@app.on_event("startup")
def on_startup():
    """Create the database tables on startup."""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created.")

@app.post("/feed-tweet")
def feed_tweet(tweet_input: TweetInput, db: Session = Depends(get_db)):
    """
    Receives a new tweet, analyzes sentiment, and stores it in the database.
    This is called by the data_feeder.py script.
    """
    if not sentiment_pipeline:
        raise HTTPException(status_code=500, detail="Sentiment model not loaded")

    try:
        # 1. Analyze sentiment
        result = sentiment_pipeline(tweet_input.text)[0]
        sentiment_label = result['label']
        sentiment_score = result['score']
        
        # 2. Create database entry
        db_tweet = Tweet(
            text=tweet_input.text,
            sentiment_label=sentiment_label.upper(),
            sentiment_score=sentiment_score
        )
        
        # 3. Add to session and commit
        db.add(db_tweet)
        db.commit()
        db.refresh(db_tweet)
        
        print(f"Processing tweet: {sentiment_label} - {db_tweet.text[:50]}...")
        
        return {
            "id": db_tweet.id,
            "text": db_tweet.text,
            "sentiment_label": db_tweet.sentiment_label,
            "sentiment_score": db_tweet.sentiment_score,
            "timestamp": db_tweet.timestamp
        }

    except Exception as e:
        print(f"Error processing tweet: {e}")
        raise HTTPException(status_code=500, detail="Error processing tweet")

@app.get("/dashboard-data")
def get_dashboard_data(db: Session = Depends(get_db)):
    """
    Aggregates data for the React dashboard.
    This is called by the React frontend every few seconds.
    """
    try:
        # Define the time window (e.g., last 10 minutes)
        time_window = datetime.datetime.utcnow() - datetime.timedelta(minutes=10)
        
        # 1. Get recent tweets for the live feed
        recent_tweets_query = (
            select(Tweet)
            .where(Tweet.timestamp >= time_window)
            .order_by(Tweet.timestamp.desc())
            .limit(10)
        )
        recent_tweets = db.execute(recent_tweets_query).scalars().all()
        
        # 2. Get sentiment counts (overall)
        total_counts_query = (
            select(Tweet.sentiment_label, func.count(Tweet.sentiment_label))
            .group_by(Tweet.sentiment_label)
        )
        total_counts_result = db.execute(total_counts_query).all()
        total_counts = {label: count for label, count in total_counts_result}

        # 3. Get sentiment counts (in the time window)
        window_counts_query = (
            select(Tweet.sentiment_label, func.count(Tweet.sentiment_label))
            .where(Tweet.timestamp >= time_window)
            .group_by(Tweet.sentiment_label)
        )
        window_counts_result = db.execute(window_counts_query).all()
        window_counts = {label: count for label, count in window_counts_result}
        
        # 4. Get data for time-series chart (e.g., per minute for last 10 minutes)
        time_series_data = []
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
                "name": f"{i}m ago",
                "POSITIVE": minute_counts.get("POSITIVE", 0),
                "NEGATIVE": minute_counts.get("NEGATIVE", 0),
            })
            
        # 5. Get data for word clouds
        positive_tweets_query = (
            select(Tweet.text)
            .where(Tweet.timestamp >= time_window)
            .where(Tweet.sentiment_label == "POSITIVE")
            .limit(100) # Limit to 100 recent tweets to process
        )
        positive_texts = [clean_text(row[0]) for row in db.execute(positive_tweets_query).all()]
        positive_word_cloud = get_word_frequencies(positive_texts)
        
        negative_tweets_query = (
            select(Tweet.text)
            .where(Tweet.timestamp >= time_window)
            .where(Tweet.sentiment_label == "NEGATIVE")
            .limit(100)
        )
        negative_texts = [clean_text(row[0]) for row in db.execute(negative_tweets_query).all()]
        negative_word_cloud = get_word_frequencies(negative_texts)

        return {
            "kpis": {
                "total_positive": total_counts.get("POSITIVE", 0),
                "total_negative": total_counts.get("NEGATIVE", 0),
                "window_positive": window_counts.get("POSITIVE", 0),
                "window_negative": window_counts.get("NEGATIVE", 0),
                "total_processed": sum(total_counts.values())
            },
            "time_series": time_series_data,
            "recent_tweets": recent_tweets,
            "word_clouds": {
                "positive": positive_word_cloud,
                "negative": negative_word_cloud
            }
        }
    except Exception as e:
        print(f"Error in /dashboard-data: {e}")
        raise HTTPException(status_code=500, detail="Error fetching dashboard data")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)