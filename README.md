# Cluster questions with LLM

This project processes clusters of questions by grouping them, extracting labels, and generating short, meaningful descriptions for each cluster using an LLM call (via ollama).

## Overview

- **Cluster Processing:** Groups questions into clusters.
- **Cluster Summarization:** Counts the questions, samples some questions, and generates a concise description for each cluster.
- **Output:** A JSON file (`clustered_questions.json`) that includes:
  - `name`: The cluster name.
  - `description`: A short, meaningful description of the cluster.
  - `count`: The number of questions in the cluster.

## Requirements

- Python 3.8+
- ollama (LLM service must be installed and accessible)
- Install required packages
    ```bash
    pip install -r requirements.txt

## How to Run

1. **Configure:**  
   Adjust any necessary settings in `config.py`.

2. **Cluster Questions:**  
   Run the clustering script:
   ```bash
   python cluster_questions.py
   
3. **Generate Summaries:**  
   Execute the summarization script:
   ```bash
   python summarize_clusters.py
   
4. **Review Output:**  
   Check the clustered_questions.json file for the summary of clusters.
5. 