# veloria-tech-ml-intern-assignment

## Project Overview

This project demonstrates an end-to-end AI/ML pipeline consisting of:

1. Web Scraping of IPL cricket match data
2. Machine Learning model for match winner prediction
3. Semantic Search system using Vector Embeddings and ChromaDB (RAG)

The project showcases Python development, data engineering, machine learning, and modern AI retrieval techniques.

---

# Assignment Objectives

### Task 1 – Web Scraping

Collect cricket match data and store it in a structured CSV format.

Collected Fields:

* Match Date
* Team 1
* Team 2
* Venue
* Match Winner
* Top Scorer
* Top Scorer Runs

Output:

```text
match_data.csv
```

---

### Task 2 – Machine Learning Prediction

Build ML models to predict the winning team.

Algorithms Implemented:

* Logistic Regression
* Random Forest

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

Output Files:

```text
best_model.pkl
confusion_matrix.png
model_comparison.csv
```

---

### Task 3 – Semantic Search (RAG)

Implemented a semantic search system using:

* Sentence Transformers
* all-MiniLM-L6-v2
* ChromaDB

Users can search cricket matches using natural language queries.

Example:

```text
Show matches where Chennai Super Kings won
```

---

# Technologies Used

## Programming Language

* Python 3.12

## Libraries

* pandas
* numpy
* requests
* beautifulsoup4
* scikit-learn
* matplotlib
* joblib
* sentence-transformers
* chromadb

---

# Project Structure

```text
veloria-tech-ml-intern-assignment/
│
├── scraper.py
├── model.py
├── rag_search.py
├── match_data.csv
├── README.md
├── requirements.txt
│
├── output/
│   ├── best_model.pkl
│   ├── confusion_matrix.png
│   ├── model_comparison.csv
│
├── screenshots/
│
└── chroma_db/
```

---

# Installation

Clone repository:

```bash
git clone <repository-url>
cd veloria-tech-ml-intern-assignment
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running Task 1

```bash
python scraper.py
```

Output:

```text
match_data.csv
```

---

# Running Task 2

```bash
python model.py
```

Generated Files:

```text
best_model.pkl
confusion_matrix.png
model_comparison.csv
```

---

# Running Task 3

```bash
python rag_search.py
```

Example Queries:

```text
Show matches where Chennai Super Kings won

Show matches played at Chennai

Find games where top scorer scored 100 runs

Show matches against Mumbai Indians
```

---

# Model Results

| Model               | Accuracy | F1 Score |
| ------------------- | -------- | -------- |
| Logistic Regression | 33.33%   | 0.3333   |
| Random Forest       | 33.33%   | 0.1667   |

Selected Model:

```text
Logistic Regression
```

Reason:

Higher F1 Score and better generalization on the available dataset.

---

# Semantic Search Results

The RAG system successfully retrieves the top 3 most relevant matches for a user query using vector similarity search.

Example Query:

```text
Show matches where Chennai Super Kings won
```

Returns:

* Most relevant CSK victories
* Match details
* Venue
* Top scorer information

---

# Challenges Faced

1. Extracting structured cricket data from HTML pages.
2. Parsing scorecard information reliably.
3. Handling small dataset size.
4. Creating meaningful features for prediction.
5. Understanding vector embeddings and semantic search.
6. Integrating Sentence Transformers with ChromaDB.

---

# Future Improvements

1. Collect 100+ IPL matches.
2. Add player-level statistics.
3. Include toss information.
4. Add weather conditions.
5. Use XGBoost and LightGBM.
6. Deploy as a web application.
7. Connect RAG system to an LLM for complete Retrieval-Augmented Generation.

---

# Screenshots

Add screenshots of:

1. Scraper execution
2. CSV output
3. Model output
4. Confusion Matrix
5. Feature Importance
6. RAG Search Results

Store screenshots inside:

```text
screenshots/
```

---

# Conclusion

This project demonstrates practical experience in:

* Web Scraping
* Data Processing
* Machine Learning
* Model Evaluation
* Embeddings
* Vector Databases
* Retrieval-Augmented Generation (RAG)

The solution provides a complete AI/ML workflow from data collection to intelligent semantic search.
