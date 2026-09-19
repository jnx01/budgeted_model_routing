"""
run_calibration.py — Measures how much the large model helps, per category.

WHAT THIS SCRIPT DOES:
  1. Loads the 70 validation questions (the calibration set).
  2. Runs every question through the SMALL model, saving its answer.
  3. Unloads the small model, loads the LARGE model (to save memory),
     and runs every question through it too.
  4. Saves all results to results/calibration.csv.
  5. Builds the category -> gain table and saves it to
     results/category_gains.csv, and prints it so we can see where the
     large model seems to help most.

The output of this script is what the
category-gain router uses later to make its decisions.

Run it with:  python run_calibration.py
"""

import os

import pandas as pd

from src.data import load_calibration_data
from src.inference import ModelRunner, SMALL_MODEL_PATH, LARGE_MODEL_PATH
from src.router import build_gain_table

# Where to save the results.
RESULTS_DIR = "results"
CALIBRATION_CSV = os.path.join(RESULTS_DIR, "calibration.csv")
GAIN_TABLE_CSV = os.path.join(RESULTS_DIR, "category_gains.csv")


def run_model_on_questions(model_path, questions, model_nickname):
    """
    Run one model over a list of questions.

    Returns a dictionary: question_id -> (predicted_letter, seconds_taken).

    The model is loaded at the start and unloaded at the end, so only one
    model is ever in memory at a time (the spec's memory-safety rule).
    """
    print(f"Loading {model_nickname} model from {model_path} ...")
    runner = ModelRunner(model_path)

    answers = {}
    total = len(questions)
    for i, question in enumerate(questions, start=1):
        letter, seconds = runner.predict(question)
        answers[question["question_id"]] = (letter, seconds)

        # Print progress every 10 questions so we can see it is working.
        if i % 10 == 0 or i == total:
            print(f"  {model_nickname}: {i}/{total} questions done")

    runner.unload()
    print(f"  {model_nickname} model unloaded.")
    return answers


def main():
    # Make sure the results folder exists.
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Step 1: load the 70 calibration questions.
    print("Loading calibration questions (70 validation questions)...")
    questions = load_calibration_data()
    print(f"  Loaded {len(questions)} questions.\n")

    # Step 2: run the small model on all questions.
    small_answers = run_model_on_questions(SMALL_MODEL_PATH, questions, "small (1.7B)")
    print()

    # Step 3: run the large model on all questions.
    large_answers = run_model_on_questions(LARGE_MODEL_PATH, questions, "large (4B)")
    print()

    # Step 4: combine everything into one row per question and save.
    rows = []
    for question in questions:
        qid = question["question_id"]
        small_letter, small_time = small_answers[qid]
        large_letter, large_time = large_answers[qid]
        rows.append({
            "question_id": qid,
            "category": question["category"],
            "correct_answer": question["answer"],
            "small_prediction": small_letter,
            "small_correct": small_letter == question["answer"],
            "small_time": small_time,
            "large_prediction": large_letter,
            "large_correct": large_letter == question["answer"],
            "large_time": large_time,
        })

    calibration_df = pd.DataFrame(rows)
    calibration_df.to_csv(CALIBRATION_CSV, index=False)
    print(f"Saved per-question results to {CALIBRATION_CSV}")

    # Step 5: build and save the category -> gain table.
    gain_table = build_gain_table(rows)
    gain_table.to_csv(GAIN_TABLE_CSV, index=False)
    print(f"Saved category gain table to {GAIN_TABLE_CSV}\n")

    # Print the table so we can see the results right away.
    # Accuracies are shown as percentages with one decimal place.
    print("=== Category gain table (calibration set, 5 questions per category) ===")
    display = gain_table.copy()
    display["small_accuracy"] = (display["small_accuracy"] * 100).round(1)
    display["large_accuracy"] = (display["large_accuracy"] * 100).round(1)
    display["gain"] = (display["gain"] * 100).round(1)
    print(display.to_string(index=False))

    # Print overall numbers too.
    overall_small = calibration_df["small_correct"].mean() * 100
    overall_large = calibration_df["large_correct"].mean() * 100
    print()
    print(f"Overall small-model accuracy: {overall_small:.1f}%")
    print(f"Overall large-model accuracy: {overall_large:.1f}%")
    print(f"Total small-model time: {calibration_df['small_time'].sum():.1f}s")
    print(f"Total large-model time: {calibration_df['large_time'].sum():.1f}s")


if __name__ == "__main__":
    main()
