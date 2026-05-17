"""
Train or calibrate the branch selector using example data.
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.paths import EXAMPLE_DATA_FILE, SELECTOR_MODEL_DIR, ensure_dirs, print_current_config
from src.selector.selector_model import SelectorModel
from src.preprocess.question_normalizer import normalize_question
from src.utils.io_utils import read_excel_sheet
from src.utils.logging_utils import logger
from configs.constants import FUNC_CALL_DOCUMENT, FUNC_CALL_API
from sklearn.model_selection import cross_val_score


def main():
    ensure_dirs()
    print_current_config()
    # Load example data
    questions_data = read_excel_sheet(EXAMPLE_DATA_FILE, "example_question")
    results_data = read_excel_sheet(EXAMPLE_DATA_FILE, "example_result")

    # Build question → label mapping
    id_to_question = {}
    for row in questions_data:
        qid = int(row["id"])
        id_to_question[qid] = row["fun_question"]

    questions = []
    labels = []
    for row in results_data:
        qid = int(row["id"])
        func_code = row["func_code"]
        if qid in id_to_question:
            q = normalize_question(id_to_question[qid])
            questions.append(q)
            labels.append(func_code)

    logger.info(f"Training data: {len(questions)} samples")
    logger.info(f"  call_api: {labels.count(FUNC_CALL_API)}")
    logger.info(f"  call_document: {labels.count(FUNC_CALL_DOCUMENT)}")

    # Train selector
    model = SelectorModel()

    # Cross-validation
    X = model.feature_builder.fit_transform(questions)
    import numpy as np
    y = np.array(labels)

    cv_scores = cross_val_score(model.classifier, X, y, cv=5, scoring="accuracy")
    logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Train on full data
    metrics = model.train(questions, labels)
    logger.info(f"Training metrics: {metrics}")

    # Save
    model.save(SELECTOR_MODEL_DIR)
    logger.info(f"Selector model saved to {SELECTOR_MODEL_DIR}")


if __name__ == "__main__":
    main()
