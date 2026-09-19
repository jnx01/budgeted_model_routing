"""
router.py — The brain of the project (but a very simple brain, on purpose).

This module does two things:

1. Build the CALIBRATION TABLE.
   After we run both models on the 70 validation questions, we know for
   each category how accurate the small model is and how accurate the
   large model is. The difference is called the "gain":
       gain = large_model_accuracy - small_model_accuracy
   A big positive gain means "the large model helps a lot in this category".

2. Decide WHO gets the large model (the routing policies).
   Given a budget (e.g. "you may use the large model on 25% of questions"),
   each policy picks which questions get sent to the large model:
     - always_small:  nobody gets the large model (cheap baseline)
     - random:        a random lottery, repeated many times to be fair
     - category_gain: questions from high-gain categories go first

IMPORTANT RULE (from the spec): the category ranking is built ONLY from
the 70 validation questions. Test answers are never used to decide routing.
"""

import random

import pandas as pd


def build_gain_table(calibration_rows):
    """
    Build the category -> gain table from calibration results.

    Input: a list of dictionaries, one per calibration question, each with:
      - category:      e.g. "math"
      - small_correct: True/False, did the small model get it right?
      - large_correct: True/False, did the large model get it right?

    Output: a pandas DataFrame with one row per category:
      - category, small_accuracy, large_accuracy, gain
    sorted by gain, highest first.
    """
    # Put the per-question results into a table so pandas can group them.
    df = pd.DataFrame(calibration_rows)

    # Group all questions by category and average the True/False values.
    # Averaging True/False values gives the accuracy (True counts as 1).
    table = df.groupby("category").agg(
        num_questions=("category", "size"),
        small_accuracy=("small_correct", "mean"),
        large_accuracy=("large_correct", "mean"),
    ).reset_index()

    # Gain = how much the large model adds, in accuracy points.
    table["gain"] = table["large_accuracy"] - table["small_accuracy"]

    # Sort so the most promising categories come first.
    table = table.sort_values("gain", ascending=False).reset_index(drop=True)

    return table


def route_always_small(test_questions, budget_fraction):
    """
    Policy A — Always Small: nobody gets the large model.

    This is the cheap baseline. Whatever the budget is, we ignore it
    and send everything to the small model.

    Returns a set of question_ids that should use the large model
    (in this policy, the set is always empty).
    """
    return set()


def route_random(test_questions, budget_fraction, num_trials=500, seed=42):
    """
    Policy B — Random routing: a random lottery for the large model.

    For the given budget, we randomly pick that many questions to send
    to the large model. Because one random pick might be lucky or unlucky,
    we repeat the lottery num_trials times (the spec says 500) and the
    caller averages the accuracy across all trials.

    Returns a list of sets — one set of question_ids per trial.
    """
    # How many questions may use the large model?
    budget_count = int(len(test_questions) * budget_fraction)

    # All question IDs that we can pick from.
    all_ids = [q["question_id"] for q in test_questions]

    # Use our own random generator with a fixed seed so the trials are
    # reproducible (same trials every time we run the experiment).
    rng = random.Random(seed)

    trials = []
    for _ in range(num_trials):
        # Randomly choose budget_count question IDs, without repeats.
        chosen = set(rng.sample(all_ids, budget_count))
        trials.append(chosen)

    return trials


def route_category_gain(test_questions, budget_fraction, gain_table):
    """
    Policy C — Category-Gain routing: spend the budget where it helps most.

    The plan:
      1. Rank categories by their calibration gain (best first).
      2. Give the large model to ALL test questions in the best category,
         then the next best, and so on, until the budget runs out.
      3. If a category only partly fits in the remaining budget, fill the
         last spots with that category's questions (in a fixed order).

    Returns a set of question_ids that should use the large model.
    """
    budget_count = int(len(test_questions) * budget_fraction)

    # Group the test question IDs by category, so we can grab whole
    # categories at a time. We sort each category's questions by ID so
    # the choice within a category is fixed and reproducible.
    ids_by_category = {}
    for q in test_questions:
        ids_by_category.setdefault(q["category"], []).append(q["question_id"])
    for category in ids_by_category:
        ids_by_category[category].sort()

    # Walk the categories from highest gain to lowest, filling the budget.
    chosen = set()
    for category in gain_table["category"]:
        # Categories not seen in calibration get skipped (shouldn't happen,
        # since both sets cover the same 14 categories).
        if category not in ids_by_category:
            continue

        for question_id in ids_by_category[category]:
            if len(chosen) >= budget_count:
                # Budget is full — stop.
                return chosen
            chosen.add(question_id)

    return chosen
