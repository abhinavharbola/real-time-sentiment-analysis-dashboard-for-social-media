# Real-Time Sentiment Dashboard

This project implements a **real-time sentiment analysis system** with an **end-to-end data pipeline** that processes simulated social media posts and visualizes live sentiment analytics on a **Streamlit dashboard**.

The entire architecture is built in **Python**, integrating **FastAPI**, **Hugging Face Transformers**, **SQLite**, and **Streamlit** — no cloud services or GPU required.

---

## System Overview

| Component | Purpose | Technology |
|------------|----------|-------------|
| **Data Ingestion** | Simulates a live tweet stream | FastAPI + Custom Feeder |
| **ML Modeling** | Sentiment analysis | Hugging Face Transformers |
| **Data Storage** | Persistent local storage | SQLite |
| **Dashboard** | Real-time visualization | Streamlit |

---

## Project Architecture

This system is composed of **three independent Python scripts**, each running in its own terminal.

```bash
.
├── sentiment_data.csv          # Dataset (Sentiment140 - renamed)
├── backend_api.py              # FastAPI backend for processing tweets
├── data_feeder.py              # Simulated tweet stream sender
├── sentiment_dashboard.py      # Streamlit dashboard app
├── sentiment.db                # SQLite database (auto-created)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## How It Works

### `data_feeder.py` — The Stream Simulator
- Reads from the **Sentiment140** dataset.
- Randomly selects a tweet every **1–3 seconds**.
- Sends it via an HTTP POST request to the **FastAPI backend**.

### `backend_api.py` — The Processing Engine
- A **FastAPI** server that listens for incoming tweets.
- Loads a **Hugging Face sentiment model**:  
  `distilbert-base-uncased-finetuned-sst-2-english`
- Processes each tweet → predicts **Positive / Negative** sentiment + confidence score.
- Saves results to **SQLite (sentiment.db)** with timestamp.

### `sentiment_dashboard.py` — The Interactive Dashboard
- A **Streamlit** web application connected to `sentiment.db`.
- Auto-refreshes every **3 seconds** to display live updates.
- Shows:
  - Key Performance Indicators (KPIs)
  - Line and bar charts
  - Word clouds
  - Live feed of latest tweets

---

## Prerequisites

Ensure all scripts and the `requirements.txt` file are in the same directory.

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Download the Dataset
Download the **Sentiment140 dataset** from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140) or [Stanford’s page](http://help.sentiment140.com/for-students).

Rename the downloaded file to:

```bash
sentiment_data.csv
```

and place it in the project directory.

---

## Step-by-Step Execution

You’ll need **three terminals** to run the complete system.

### Terminal 1 — Run the Backend API

Start the FastAPI server:

```bash
uvicorn backend_api:app --reload
```

Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

---

### Terminal 2 — Run the Data Feeder

In a new terminal, run:

```bash
python data_feeder.py
```

Example logs:
```
Sending tweet: "Just watched an amazing movie!"
-> Response: 200
```

Leave this terminal running.

---

### Terminal 3 — Run the Streamlit Dashboard

In the third terminal, start the dashboard:

```bash
streamlit run sentiment_dashboard.py
```

The dashboard will open automatically in your default browser at:

```
http://localhost:8501
```

At first, it will show **“Waiting for data...”** — within 10–15 seconds, it will begin displaying **live sentiment analytics** as tweets flow in.

---

## Core Technologies

| Layer | Tool / Library |
|-------|----------------|
| **API** | FastAPI |
| **Model** | Hugging Face Transformers (DistilBERT) |
| **Database** | SQLite |
| **Dashboard** | Streamlit |
| **Language** | Python 3.9+ |

---

## Author
Developed by **Abhinav Harbola**  
Data Engineering | NLP | Real-Time Systems