"""
inference.py — Runs questions through our two local models.

This module handles everything about talking to a model:
  1. Loading a model into memory (and unloading it to save memory).
  2. Asking one question and getting back the predicted answer letter.
  3. Measuring how long each question takes.

HOW ANSWERING WORKS (the "logits" method):
  We do NOT ask the model to write text. Qwen3 is a "reasoning" model
  that wants to think out loud for hundreds of tokens, which is slow
  and hard to parse. Instead, we run the prompt through the model ONCE
  and look at the model's raw scores (called "logits") for what the very
  next token should be. We compare only the 10 answer letters (A-J) and
  pick the letter with the highest score. This is:
    - fast: one forward pass per question
    - deterministic: same question always gives the same answer
    - standard: this is how MMLU-Pro is usually evaluated
"""

import time

import mlx.core as mx
from mlx_lm import load

from src.data import ANSWER_LETTERS, build_prompt

# Where our two models live on disk (both are gitignored folders).
SMALL_MODEL_PATH = "models/Qwen3-1.7B-4bit"
LARGE_MODEL_PATH = "models/Qwen3-4B-4bit"


class ModelRunner:
    """
    A wrapper around one loaded model that can answer questions.

    Usage:
        runner = ModelRunner("models/Qwen3-1.7B-4bit")
        letter, seconds = runner.predict(question)
        runner.unload()
    """

    def __init__(self, model_path):
        """
        Load the model and tokenizer from disk into memory.
        This takes a few seconds and a few GB of RAM.
        """
        self.model_path = model_path
        self.model, self.tokenizer = load(model_path)

        # Pre-compute the token ID for each answer letter (" A", " B", ...).
        # A token ID is the number the model uses internally for a piece of
        # text. We need these IDs to look up each letter's score later.
        # We use the letter with a leading space because that is how the
        # letter would naturally appear after the prompt text.
        self.letter_token_ids = []
        for letter in ANSWER_LETTERS:
            token_ids = self.tokenizer.encode(" " + letter, add_special_tokens=False)
            # Take the last token ID, in case a letter splits into pieces
            # (for these models each " A".." J" is a single token).
            self.letter_token_ids.append(token_ids[-1])

    def predict(self, question):
        """
        Ask the model one multiple-choice question.

        Returns a pair: (predicted_letter, seconds_taken).
          - predicted_letter: one of "A".."J", the model's answer
          - seconds_taken:    how long the forward pass took (float)
        """
        # Build the fixed prompt and format it as a chat message.
        # enable_thinking=False tells Qwen3 to skip its long <think> block,
        # since we only need the answer letter scores, not reasoning text.
        prompt_text = build_prompt(question)
        messages = [{"role": "user", "content": prompt_text}]
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # Turn the text into token IDs (numbers the model understands).
        token_ids = mx.array([self.tokenizer.encode(formatted)])

        # Time the forward pass: run the prompt through the model once.
        start_time = time.perf_counter()
        logits = self.model(token_ids)
        # logits has shape (1, prompt_length, vocabulary_size).
        # We only care about the LAST position: the scores for what token
        # comes next, right after the prompt.
        next_token_logits = logits[0, -1, :]
        # Force the computation to finish before we stop the timer.
        # (MLX runs computations lazily, so without this the timer would
        # stop before the real work is done.)
        mx.eval(next_token_logits)
        seconds_taken = time.perf_counter() - start_time

        # Compare the scores of just the 10 answer letters and pick the best.
        best_letter = None
        best_score = float("-inf")
        for letter, token_id in zip(ANSWER_LETTERS, self.letter_token_ids):
            score = next_token_logits[token_id].item()
            if score > best_score:
                best_score = score
                best_letter = letter

        return best_letter, seconds_taken

    def unload(self):
        """
        Remove the model from memory.

        The spec says: if the Mac runs low on memory, do not keep both
        models loaded at once. Call this when done with a model, then
        load the other one.
        """
        self.model = None
        self.tokenizer = None
        # Ask MLX to actually free the memory back to the system.
        mx.clear_cache()
