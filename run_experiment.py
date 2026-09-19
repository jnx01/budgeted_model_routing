"""
run_experiment.py — The main experiment: does smart routing beat random?

WHAT THIS SCRIPT DOES:
  1. Loads the 140 test questions.
  2. Runs both models on all of them and saves every prediction to
     results/predictions.csv. (This is the only part that uses the models.
     Everything after this just reuses the saved answers.)
  3. Loads the category gain table built during calibration.
  4. Evaluates three routing policies at five budgets (10/25/50/75/100%):
       - always small   (cheap baseline)
       - random         (500 random lotteries per budget, averaged)
       - category-gain  (spend the budget on the highest-gain categories)
  5. Saves the summary table and makes the three plots from the spec:
       - Plot 1: calibration gain by category (bar chart)
       - Plot 2: accuracy vs large-model budget (line chart)
       - Plot 3: budget/accuracy summary table (CSV + printed)

Run it with:  python run_experiment.py
"""

import os

import matplotlib

# Use a backend that saves to files instead of opening windows.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data import load_test_data
from src.evaluate import evaluate_policy, evaluate_random_policy
from src.inference import LARGE_MODEL_PATH, SMALL_MODEL_PATH, ModelRunner
from src.router import (
    route_always_small,
    route_category_gain,
    route_random,
)
from run_calibration import run_model_on_questions

# Where to save everything.
RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
PREDICTIONS_CSV = os.path.join(RESULTS_DIR, "predictions.csv")
GAIN_TABLE_CSV = os.path.join(RESULTS_DIR, "category_gains.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "budget_summary.csv")

# The budgets to test: what fraction of questions may use the large model.
BUDGETS = [0.10, 0.25, 0.50, 0.75, 1.00]

# How many random lotteries per budget for the random policy (spec says 500).
RANDOM_TRIALS = 500


def collect_test_predictions(questions):
    """
    Run both models on all test questions and save the predictions.

    If results/predictions.csv already exists, we load it instead of
    re-running the models. This saves time when we re-run the analysis.

    Returns the list of per-question result dictionaries.
    """
    if os.path.exists(PREDICTIONS_CSV):
        print(f"Found existing {PREDICTIONS_CSV} — loading saved predictions.")
        print("  (Delete that file if you want to re-run the models.)\n")
        return pd.read_csv(PREDICTIONS_CSV).to_dict("records")

    print("Running both models on all test questions...")
    small_answers = run_model_on_questions(SMALL_MODEL_PATH, questions, "small (1.7B)")
    print()
    large_answers = run_model_on_questions(LARGE_MODEL_PATH, questions, "large (4B)")
    print()

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

    pd.DataFrame(rows).to_csv(PREDICTIONS_CSV, index=False)
    print(f"Saved test predictions to {PREDICTIONS_CSV}\n")
    return rows


def make_gain_plot(gain_table):
    """
    Plot 1 — Calibration gain by category (bar chart).

    One bar per category: how many accuracy points the large model added
    on the calibration questions. This is the evidence the router uses.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    categories = gain_table["category"]
    gains_percent = gain_table["gain"] * 100  # show as percentage points

    # Green bars for positive gain, red bars for negative gain.
    colors = ["#2ecc71" if g >= 0 else "#e74c3c" for g in gains_percent]
    ax.bar(categories, gains_percent, color=colors)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Category")
    ax.set_ylabel("Large-model gain (accuracy points)")
    ax.set_title("Where does the large model help? (calibration set, 5 questions per category)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "plot1_calibration_gain.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def make_budget_plot(summary_rows):
    """
    Plot 2 — Accuracy vs large-model budget (line chart).

    Three lines: always-small (flat baseline), random, and category-gain.
    This is the main figure of the project.
    """
    df = pd.DataFrame(summary_rows)

    fig, ax = plt.subplots(figsize=(8, 5))

    budgets_percent = df["budget"] * 100

    # Always-small is a flat line (it never uses the large model).
    ax.plot(budgets_percent, df["always_small_accuracy"] * 100,
            marker="s", label="Always small", color="#95a5a6", linestyle="--")
    ax.plot(budgets_percent, df["random_accuracy"] * 100,
            marker="o", label="Random (mean of 500 trials)", color="#3498db")
    ax.plot(budgets_percent, df["category_gain_accuracy"] * 100,
            marker="^", label="Category-gain", color="#e67e22")

    ax.set_xlabel("Large-model budget (% of questions)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy vs large-model budget (140 test questions)")
    ax.set_xticks(budgets_percent)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "plot2_accuracy_vs_budget.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Step 1: load the 140 test questions.
    print("Loading test questions (140 questions, 10 per category)...")
    questions = load_test_data()
    print(f"  Loaded {len(questions)} questions.\n")

    # Step 2: get both models' predictions on every test question.
    predictions = collect_test_predictions(questions)

    # Step 3: load the gain table built during calibration.
    # IMPORTANT: this table comes ONLY from the 70 validation questions.
    # No test answers were used to build it.
    gain_table = pd.read_csv(GAIN_TABLE_CSV)
    print(f"Loaded gain table from {GAIN_TABLE_CSV} "
          f"({len(gain_table)} categories).\n")

    # Step 4: evaluate the three policies at each budget.
    print("Evaluating routing policies...")
    summary_rows = []
    for budget in BUDGETS:
        # Policy A: always small (same answer at every budget).
        always_small = evaluate_policy(
            predictions, route_always_small(questions, budget))

        # Policy B: random, averaged over 500 lotteries.
        trials = route_random(questions, budget, num_trials=RANDOM_TRIALS)
        random_result = evaluate_random_policy(predictions, trials)

        # Policy C: category-gain routing.
        category_ids = route_category_gain(questions, budget, gain_table)
        category_result = evaluate_policy(predictions, category_ids)

        summary_rows.append({
            "budget": budget,
            "large_calls": category_result["large_calls"],
            "always_small_accuracy": always_small["accuracy"],
            "random_accuracy": random_result["mean_accuracy"],
            "random_std": random_result["std_accuracy"],
            "category_gain_accuracy": category_result["accuracy"],
            "category_gain_time_seconds": category_result["total_time_seconds"],
        })

        print(f"  Budget {budget:4.0%}: "
              f"always-small {always_small['accuracy']:.1%} | "
              f"random {random_result['mean_accuracy']:.1%} | "
              f"category-gain {category_result['accuracy']:.1%}")

    # Step 5: save the summary table (Plot 3 from the spec).
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved summary table to {SUMMARY_CSV}\n")

    print("=== Budget / accuracy summary ===")
    display = summary_df.copy()
    for col in ["always_small_accuracy", "random_accuracy", "category_gain_accuracy"]:
        display[col] = (display[col] * 100).round(1)
    display["budget"] = (display["budget"] * 100).round(0).astype(int).astype(str) + "%"
    print(display.to_string(index=False))
    print()

    # Step 6: make the two plots.
    make_gain_plot(gain_table)
    make_budget_plot(summary_rows)

    # Step 7: print the timing comparison (Experiment 4 from the spec).
    predictions_df = pd.DataFrame(predictions)
    always_large_time = predictions_df["large_time"].sum()
    always_small_time = predictions_df["small_time"].sum()
    print()
    print("=== Measured inference time ===")
    print(f"Always small: {always_small_time:.1f}s")
    print(f"Always large: {always_large_time:.1f}s")
    for row in summary_rows:
        print(f"Category-gain at {row['budget']:4.0%} budget: "
              f"{row['category_gain_time_seconds']:.1f}s")


if __name__ == "__main__":
    main()
