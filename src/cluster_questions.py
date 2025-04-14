import csv
import json
import multiprocessing
import os
from functools import partial
import random

from ollama import chat
from loguru import logger

from prompt_templates.utils import read_prompt_template
from src import CFG


# Configuration
CSV_FILE = CFG.data_file_path
OUTPUT_JSON = CFG.output_json_path
LLM_MODEL = 'llama3.2'
CANDIDATE_SAMPLE_SIZE = 500  # Number of questions to sample for candidate label extraction


def load_questions():
    """
    Load questions from the CSV and return a list.
    """
    questions = []
    if not os.path.exists(CSV_FILE):
        print(f"CSV file '{CSV_FILE}' not found.")
        return questions

    with open(CSV_FILE, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            q = row.get('instruction', '').strip()
            if q:
                questions.append(q)
    print(f"Total questions loaded: {len(questions)}")
    return questions

def build_candidate_prompt(questions):
    """
    Constructs a prompt to extract candidate semantic labels.
    The output should be a JSON object with key "labels" mapping to a list of candidate labels.
    """
    formatted_questions = "\n".join([f"Question: {q.strip()}" for q in questions])
    prompt = read_prompt_template("label_extraction").render(questions=formatted_questions)
    return prompt

def extract_candidate_labels(questions):
    """
    Uses ollama.chat to extract candidate labels from a sample of questions.
    """
    prompt = build_candidate_prompt(questions)
    try:
        response = chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"temperature": 0.0}
        )
        result = json.loads(response['message']["content"])
        labels = result.get('labels', [])
        logger.info(f"Candidate labels extracted: {labels}")
        with open(CFG.labels_path, "w") as f:
            f.write("\n".join(labels))
        return labels
    except Exception as e:
        print(f"Error extracting candidate labels: {e}")
        return []

def build_classification_prompt(question, labels):
    """
    Constructs a prompt for classifying a batch of questions using the candidate labels.
    The LLM must choose the best label for each question from the provided set.
    The expected output is a JSON array of objects, each with keys 'question' and 'cluster'.
    """
    return read_prompt_template("question_classification").render(labels=labels, question=question)


def classify_question(question, labels):
    """
    Classify a single question using the candidate label set.
    Returns a dictionary with the question and its assigned cluster label.
    """
    prompt = build_classification_prompt(question, labels)
    try:
        response = chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
    except Exception as e:
        logger.error(f"Error during LLM call for question '{question}': {e}")
        return {"question": question, "cluster": "error"}

    try:
        result = json.loads(response['message']["content"])
        if isinstance(result, dict) and "label" in result:
            assert result["label"] in labels, f"Unknown label, got {result['label']}"

            return {"question": question, "cluster": result["label"]}
        else:
            raise ValueError("Invalid response format")
    except Exception as e:
        logger.error(f"Error parsing JSON response for question '{question}': {e}")
        return {"question": question, "cluster": "unknown"}


def process_question(question, labels):
    """
    Wrapper function to classify a single question.
    """
    return classify_question(question, labels)


def main():
    questions = load_questions()
    if not questions:
        logger.warning("No questions loaded. Exiting.")
        return

    # Sample a subset of questions to extract candidate labels.
    sample_questions = random.sample(questions, min(CANDIDATE_SAMPLE_SIZE, len(questions)))
    if not CFG.labels_path.exists():
        labels = extract_candidate_labels(sample_questions)
    else:
        with open(CFG.labels_path) as f:
            labels = f.read().split("\n")
    if not labels:
        logger.error("No candidate labels extracted. Aborting classification.")
        return

    os.makedirs(CFG.clusters_dir, exist_ok=True)

    cluster_files = {}

    process_question_partial = partial(process_question, labels=labels)

    processed_count = 0

    # fixme: there is a little bug (some questions are processed more than once) with multiprocessing, fix later
    with multiprocessing.Pool(os.cpu_count() * 2) as pool:
        for result in pool.imap_unordered(process_question_partial, questions, chunksize=100):
            cluster_label = result.get("cluster", "unknown")
            if cluster_label not in cluster_files:
                file_path = os.path.join(CFG.clusters_dir, f"{cluster_label}.jsonl")
                cluster_files[cluster_label] = open(file_path, "a", encoding="utf-8")
                processed_count += 1
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count} questions so far.")
            cluster_files[cluster_label].write(json.dumps(result) + "\n")

    for fh in cluster_files.values():
        fh.close()

    logger.info(f"Total questions processed: {processed_count}")
    logger.success("Classification complete.")


if __name__ == "__main__":
    main()