import pandas as pd
import requests
import time
import random
import os

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/feed-tweet"
DATASET_PATH = "sentiment140.csv" 
# Download from Kaggle: "Sentiment140 dataset with 1.6 million tweets"
# Make sure the CSV file is in the same directory as this script.
# Expected columns: 'target', 'ids', 'date', 'flag', 'user', 'text'

def check_dataset():
    """Checks if the dataset file exists."""
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset file not found at '{DATASET_PATH}'")
        print("Please download the 'Sentiment140' dataset from Kaggle.")
        print("Rename the file to 'sentiment140.csv' and place it in this directory.")
        return False
    return True

def load_dataset():
    """Loads the dataset into a pandas DataFrame."""
    try:
        # Adjust column names as per the dataset
        col_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
        df = pd.read_csv(
            DATASET_PATH, 
            encoding='latin-1', 
            header=None, 
            names=col_names
        )
        
        # We only need the 'text' column
        texts = df['text'].tolist()
        print(f"Loaded {len(texts)} tweets from '{DATASET_PATH}'.")
        return texts
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def start_streaming(tweet_texts):
    """Starts the real-time simulation loop."""
    print(f"Starting to stream data to {API_URL}...")
    print("Press Ctrl+C to stop.")
    
    while True:
        try:
            # 1. Pick a random tweet
            tweet_text = random.choice(tweet_texts)
            
            # 2. Send it to the API
            payload = {"text": tweet_text}
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                print(f"Sent tweet: {tweet_text[:60]}...")
            else:
                print(f"Error sending tweet. Status: {response.status_code}, Body: {response.text}")

            # 3. Wait for a random time (simulates real-world jitter)
            sleep_time = random.uniform(1.0, 3.0)
            time.sleep(sleep_time)
            
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to backend API. Is it running?")
            print("Retrying in 10 seconds...")
            time.sleep(10)
        except KeyboardInterrupt:
            print("\nStopping data feeder.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            time.sleep(5)

if __name__ == "__main__":
    if check_dataset():
        dataset = load_dataset()
        if dataset:
            start_streaming(dataset)