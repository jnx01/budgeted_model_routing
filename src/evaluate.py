"""
evaluate.py — Measures how well a routing policy did.

This module takes the saved predictions (which letter each model picked
for each question, and whether that was right) plus a routing decision
(which questions go to the large model) and computes the numbers we
care about:

  1. Accuracy:            how many questions did we get right overall?
  2. Large-model usage:   what fraction of questions used the large model?
  3. Improvement:         how much better is this than Always-Small?
  4. Inference time:      how long would this policy take in total?

The key idea: for each question we ALREADY have both models' predictions
saved. A routing policy just picks which model's answer to "use" for each
question. So evaluating a policy needs no new model calls — we just look
up the saved answers.
"""

import pandas as pd


def evaluate_policy(predictions, large_model_ids):
    """
    Score one routing decision.

    Inputs:
      - predictions: a list of dictionaries, one per test question, with:
          question_id, small_correct, large_correct, small_time, large_time
      - large_model_ids: a set of question_ids that should use the large
        model's answer. Every other question uses the small model's answer.

    Returns a dictionary with:
      - accuracy:            fraction of questions answered correctly
      - num_correct:         how many were correct
      - num_questions:       total questions
      - large_calls:         how many questions used the large model
      - large_usage:         fraction that used the large model
      - total_time_seconds:  estimated total inference time for this policy
    """
    num_questions = len(predictions)
    num_correct = 0
    total_time = 0.0

    for row in predictions:
        # Which model's answer does this policy use for this question?
        uses_large = row["question_id"] in large_model_ids

        if uses_large:
            is_correct = row["large_correct"]
            total_time += row["large_time"]
        else:
            is_correct = row["small_correct"]
            total_time += row["small_time"]

        if is_correct:
            num_correct += 1

    large_calls = len(large_model_ids)

    return {
        "accuracy": num_correct / num_questions,
        "num_correct": num_correct,
        "num_questions": num_questions,
        "large_calls": large_calls,
        "large_usage": large_calls / num_questions,
        "total_time_seconds": total_time,
    }


def evaluate_random_policy(predictions, trials):
    """
    Score the random policy: average the accuracy over all its trials.

    Each trial is one random pick of which questions get the large model.
    We evaluate every trial and return the mean accuracy (and the spread,
    so we can see how much luck varies).

    Returns a dictionary with:
      - mean_accuracy:  average accuracy across trials
      - std_accuracy:   standard deviation across trials (how lucky/unlucky
                        a single random pick can be)
      - min_accuracy:   worst trial
      - max_accuracy:   best trial
      - num_trials:     how many trials were run
    """
    trial_accuracies = []
    for trial_ids in trials:
        result = evaluate_policy(predictions, trial_ids)
        trial_accuracies.append(result["accuracy"])

    series = pd.Series(trial_accuracies)
    return {
        "mean_accuracy": series.mean(),
        "std_accuracy": series.std(),
        "min_accuracy": series.min(),
        "max_accuracy": series.max(),
        "num_trials": len(trials),
    }


def improvement_over_baseline(policy_accuracy, baseline_accuracy):
    """
    How much better is a policy than the Always-Small baseline?

    Returns the difference in percentage points. For example, if Always
    Small gets 55% and the policy gets 59%, this returns +4.0.
    """
    return (policy_accuracy - baseline_accuracy) * 100
