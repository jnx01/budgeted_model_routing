"""
data.py — Loads the MMLU-Pro questions for our project.

This module has one job: give us the two sets of questions we need.

1. Calibration set: all 70 validation questions.
   We use these to MEASURE how much the big model helps in each category.

2. Test set: 140 questions (10 from each of the 14 categories).
   We use these to EVALUATE our routing policies fairly.

It also builds the one fixed prompt that both models see, so the
only difference between models is the model itself.
"""

import os

# The Hugging Face "xet" download protocol kept stalling on this machine.
# Turning it off forces plain HTTP downloads, which work reliably.
# This must be set BEFORE importing datasets.
os.environ["HF_HUB_DISABLE_XET"] = "1"

from datasets import load_dataset

# The 14 categories in MMLU-Pro, and the letters for the 10 answer options.
# Every MMLU-Pro question has exactly 10 options: A through J.
ANSWER_LETTERS = "ABCDEFGHIJ"

# How many test questions to take from each category (10 x 14 = 140 total).
TEST_QUESTIONS_PER_CATEGORY = 10

# A fixed random seed so we always pick the SAME 140 test questions.
# This makes the experiment reproducible: anyone running this code
# gets exactly the same questions.
RANDOM_SEED = 42


def load_calibration_data():
    """
    Load the calibration set: all 70 validation questions from MMLU-Pro.

    Returns a list of dictionaries. Each dictionary is one question with:
      - question_id: a unique number for the question
      - question:    the question text
      - options:     a list of 10 answer choices (strings)
      - answer:      the correct answer letter, e.g. "A"
      - category:    the subject area, e.g. "math"
    """
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")

    # Convert the dataset rows into plain Python dictionaries,
    # keeping only the fields we need.
    questions = []
    for row in dataset:
        questions.append({
            "question_id": row["question_id"],
            "question": row["question"],
            "options": row["options"],
            "answer": row["answer"],
            "category": row["category"],
        })

    return questions


def load_test_data():
    """
    Load the test set: 140 questions from MMLU-Pro (10 per category).

    The full test split has 12,032 questions. We take a balanced subset:
    the first 10 questions from each of the 14 categories, after shuffling
    with a fixed seed. "Balanced" means every category gets equal weight,
    so no category can dominate the results.

    Returns the same kind of list of dictionaries as load_calibration_data().
    """
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    # Shuffle the full test set with a fixed seed so the selection is
    # random but always the same (reproducible).
    shuffled = dataset.shuffle(seed=RANDOM_SEED)

    # Walk through the shuffled questions and collect up to 10 per category.
    # We stop as soon as every category has 10.
    picked_per_category = {}   # category name -> list of questions picked
    questions = []
    for row in shuffled:
        category = row["category"]

        # How many have we already picked from this category?
        already_picked = len(picked_per_category.get(category, []))

        # Skip this question if the category is already full.
        if already_picked >= TEST_QUESTIONS_PER_CATEGORY:
            continue

        # Add the question to our test set.
        question = {
            "question_id": row["question_id"],
            "question": row["question"],
            "options": row["options"],
            "answer": row["answer"],
            "category": category,
        }
        questions.append(question)
        picked_per_category.setdefault(category, []).append(question)

    return questions


def build_prompt(question):
    """
    Build the one fixed prompt that BOTH models see for a question.

    The spec requires exactly the same prompt for both models, so the
    comparison is fair. The prompt lists the question and all 10 options,
    then asks for just the answer letter.

    Returns the prompt as a single string.
    """
    # Turn the list of options into lines like "A. <option text>".
    option_lines = []
    for i, option_text in enumerate(question["options"]):
        letter = ANSWER_LETTERS[i]
        option_lines.append(f"{letter}. {option_text}")
    options_block = "\n".join(option_lines)

    # This is the exact prompt format from the project spec.
    prompt = f"""Answer the multiple-choice question.

Question:
{question['question']}

Options:
{options_block}

Return only the letter of the correct answer."""

    return prompt
