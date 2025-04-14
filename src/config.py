from pathlib import Path

class CFG:
    parent_dir = Path(__file__).parents[1]
    data_dir = parent_dir / "data"
    clusters_dir = data_dir / "clusters"
    templates_dir = parent_dir / "prompt_templates"

    data_file_path = data_dir / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    output_json_path = data_dir / "clustered_questions.json"
    labels_path = data_dir / "labels.txt"

