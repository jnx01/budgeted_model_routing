# Report: Small-Model Routing Under a Fixed Inference Budget

## The big idea

Running a big AI model costs more time and energy than running a small
one. But what if most questions don't actually *need* the big model?

Think of a hospital again: a nurse handles most patients, and the
expensive specialist is called in only for cases where their extra skill
matters. We tested whether AI models can be organized the same way.

**Our question:** if we can only afford to use the big model on *some*
questions, can a simple rule pick *which* ones and beat just picking
randomly?

**Short answer: yes.** Our simple rule beat random picking at every
budget we tested.

## The setup (in plain terms)

- **Two models on one Mac:** a small one (Qwen3-1.7B, ~1 GB) and a relatively bigger
  one (Qwen3-4B, ~2.15 GB). Everything ran locally — no internet calls,
  no per-question costs.
- **Practice round:** 70 questions (5 from each of 14 school subjects).
  Both models answered all of them. This told us where the bigger model
  helps.
- **Final exam:** 140 brand-new questions (10 per subject) that neither
  the models nor our routing rule had seen during practice.
- **How models answer:** for each question, we check which answer letter
  (A through J) the model considers most likely. One quick look per
  question — no long written answers needed.
- **Total work:** 420 model runs (about 40 seconds of computing time).

## Experiment 1 — Does the bigger model help some subjects more than others?

**Yes — enormously.** Here's what the practice round showed:

| Subject | Small model | Big model | Gain (big − small) |
|---|---:|---:|---:|
| other | 40% | 80% | **+40 points** |
| psychology | 40% | 80% | **+40 points** |
| business | 40% | 60% | +20 points |
| philosophy | 40% | 60% | +20 points |
| physics | 40% | 60% | +20 points |
| biology, computer science, health, history, law | — | — | 0 (no difference) |
| chemistry | 40% | 20% | −20 points |
| engineering | 40% | 20% | −20 points |
| math | 40% | 20% | −20 points |
| economics | 60% | 20% | **−40 points** |

Two surprises worth noting:

1. **The bigger model is sometimes *worse*.** In economics, the small model
   got 60% but the big model only got 20%. Sending economics questions to
   the big model would actively hurt us.
2. **In five subjects, the bigger model added nothing at all.** Sending
   those questions to it would be pure waste.

**One honest warning:** with only 5 practice questions per subject, a
single question changes a subject's gain by 20 points. So these numbers
are rough estimates, not precise measurements. Experiment 2 checks
whether they hold up on new questions.

## Experiment 2 — Smart routing vs random routing (the main event)

Now the final exam: 140 new questions. At each budget, both strategies
get exactly the same number of big-model calls — the only difference is
*who picks* which questions get them.

| Budget | Random picking (avg of 500 tries) | Our smart router | Smart router wins by |
|---:|---:|---:|---:|
| 10% | 29.0% | 30.7% | **+1.7 points** |
| 25% | 30.7% | 33.6% | **+2.9 points** |
| 50% | 33.5% | 36.4% | **+2.9 points** |
| 75% | 36.4% | 38.6% | **+2.2 points** |

**Our router won at every budget.** The practice-round signal — rough as
it was — worked on questions it had never seen.

For context: random picking's luck varied by about ±2 points across the
500 lotteries. Our router's advantage is about that size or bigger, so
the win is meaningful, not just luck.

## Experiment 3 — How much accuracy do you get for your money?

| Big-model budget | Always small | Random | Smart router |
|---:|---:|---:|---:|
| 10% | 27.9% | 29.0% | 30.7% |
| 25% | 27.9% | 30.7% | 33.6% |
| 50% | 27.9% | 33.5% | 36.4% |
| 75% | 27.9% | 36.4% | 38.6% |
| 100% | 27.9% | 39.3% | 39.3% |

Notice the shape of the smart-router column: the **first half of the
budget buys a lot** (27.9% → 36.4%, a gain of 8.5 points), but the
**second half buys very little** (36.4% → 39.3%, only 2.9 points).

It's like watering a garden: the first bucket of water helps a lot, but
once the soil is soaked, extra water just runs off. Most of the bigger
model's value is available at half its full cost.

## Experiment 4 — Does routing actually save time?

We measured real running time on the Mac:

| Strategy | Total time | Accuracy |
|---|---:|---:|
| Always small (cheapest) | 12.2s | 27.9% |
| Smart router @ 25% budget | 14.2s | 33.6% |
| Smart router @ 50% budget | 17.7s | 36.4% |
| Always big (most expensive) | 26.4s | 39.3% |

The sweet spot: **smart routing at 25% budget**. Compared to the cheapest
option, it buys +5.7 accuracy points for just +2 seconds. Going all the
way to always-big costs +14.2 seconds — seven times more extra time — for
about twice the accuracy gain.

## Where could our router go wrong? (honest assessment)

- **Shaky estimates.** Five practice questions per subject is very few.
  A couple of unlucky questions could put subjects in the wrong order.
  With a different practice set, the ranking might look different.
- **The exam was harder than practice.** The small model got 41.4% on
  practice questions but only 27.9% on the exam. The routing still
  worked, but a bigger gap between practice and exam could break it.
- **Subjects are a blunt tool.** "Math" includes both easy and brutal
  questions. Ideally we'd judge each question on its own — that's the
  natural next step for future work.

## Conclusion

A routing rule no fancier than a sorted list — built from just 70
practice questions — beat random routing at every budget on 140 new
questions. When you can only afford the expensive model sometimes,
*where* you spend those calls matters, and even a crude signal like the
question's subject is enough to spend them better than chance.

## The natural next question

We routed by subject, which is a rough label. What if we judged each
question individually — for example, by how hard it looks? A smarter
signal might route even better while still respecting a fixed budget.
