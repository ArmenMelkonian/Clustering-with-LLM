import os
import json
import random

from ollama import chat

from prompt_templates.utils import read_prompt_template
from src import CFG


LLM_MODEL = 'llama3.2'


def get_random_sample_questions(file_path, sample_size=50):
    """
    Reads a JSONL file and returns a list of sample questions.
    If there are fewer than 'sample_size' questions, returns all of them.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Ensure there are lines to sample from
    sample_lines = random.sample(lines, min(sample_size, len(lines))) if lines else []
    questions = []

    for line in sample_lines:
        try:
            data = json.loads(line)
            questions.append(data.get('question', ''))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file {file_path}: {e}")

    return questions


def generate_description_from_questions(cluster_name, questions):
    """
    Generates a description summary of the given questions using a LLM.
    """
    prompt = read_prompt_template("description").render(cluster_name=cluster_name, questions=questions)

    response = chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    response = response["message"]["content"]

    return response


def process_cluster_files(directory):
    """
    Processes all JSONL files in the provided directory and returns a list of dictionaries,
    each describing one cluster with file name, description, and count of questions.
    """
    clusters = []

    for file_name in os.listdir(directory):
        if file_name.endswith('.jsonl'):
            file_path = os.path.join(directory, file_name)

            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            count = len(lines)

            # Select a random sample of questions from the file
            questions_sample = get_random_sample_questions(file_path)
            cluster_name = file_name.split(".jsonl")[0]

            description = generate_description_from_questions(cluster_name, questions_sample)

            clusters.append({
                "name": cluster_name,
                "description": description,
                "count": count
            })

    return clusters


def save_clusters_to_json(clusters, output_file):
    """
    Saves the list of cluster dictionaries to a JSON file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)
    print(f"Clusters summary saved to {output_file}")


if __name__ == "__main__":
    clusters_dir = CFG.clusters_dir
    output_file = CFG.output_json_path

    clusters = process_cluster_files(clusters_dir)
    save_clusters_to_json(clusters, output_file)