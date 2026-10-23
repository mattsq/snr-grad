# On-policy batch-control study (specified before evaluation)

## Training and data

Freeze the nearest-five-checkpoint calibration in `benchmark_batch_budget.py`,
fitted on digit-split seeds 0–1. No multiplier, neighbor count, batch-size
range, or policy threshold is selected using the evaluation runs. Each
optimizer uses its existing fixed LR (SNRMuon 0.1, AdamW 0.003). Use a
3,000-training-example cap, batch choices `[4, 8, 16, 32, 64]`, 64-example
eight-split probes every 20 optimizer steps, and a distribution change after
exactly 1,500 training examples where relevant. A step that would straddle the
change point is shortened so no batch mixes regimes. Draw training indices
from one seeded example-indexed stream per split and task, shared by all
policies. Probe indices use a separate deterministic stream; count probe
examples and probe CPU time explicitly. Randomize arm execution order within
each seed to reduce systematic CPU timing drift.

## Policies

For each optimizer and task, run fixed B=4, 8, 16, 32, 64, the development-selected constant
batch maximizing gain per CPU second, and four adaptive arms: Euclidean and
optimizer-aware sensors, each with an example-cost and a time-cost objective.
The fitted predictor selects the candidate with the highest mean measured
gain per resource among its five nearest development probe scales. Its
example objective includes amortized probe examples; its time objective
includes amortized probe time. On invalid sensor readings fall back to the
development-selected constant for that objective. Actuate the selected batch
at the next step. Do not use true shift time or validation labels to control
batch size. All arms use the same LR and model initialization within a seed.

## Evaluation

Seeds 11–15 have not been used to tune this experiment. Evaluate stationary
handwritten digits and a +3 label permutation after the change point, then
test generalization to a 90-degree image rotation with unchanged labels and
a separate shifted-target synthetic matrix regression problem. Both SNRMuon
and AdamW get all ten arms on all four tasks: 400 trajectories. Keep full
per-step trajectories and report final validation loss after 3,000 training
examples, elapsed training plus probe CPU time, and actual examples processed
including probes. Report paired seed differences to the fixed baselines and
Euclidean counterpart. A validation curve against elapsed time uses a common
time support within each task, optimizer, and seed; before a change point,
the final-regime validation target is also recorded to keep its meaning fixed.
This time curve is descriptive: changing batch changes when a policy reaches
the shift in wall time. Final sample-capped loss is the primary endpoint.

The fixed `B=8,16,32,64` arms were added after the first 240 trajectories were
inspected, to complete the attainable fixed-batch frontier. An audit then
found that the first pass mistakenly initialized adaptive arms at the fast
fixed batch for 20 steps instead of the specified B=4. That pass is excluded.
The complete 400-arm study was rerun with all policies randomized together.
No calibration parameters were changed. Because the first pass exposed
outcomes on seeds 11–15, the corrected results are now exploratory rather
than a clean untouched-seed confirmation; independent data is required for
a confirmatory claim.

A post-hoc no-probe schedule, 20 steps at B=4 then the development-selected
fast fixed batch, adds 40 diagnostic trajectories. It isolates the common
initial small-batch warmup from any subsequent sensor action; its CPU timings
were obtained in a separate pass.

Do not claim generalization from five differently split versions of the same
digits dataset alone. The rotation tests a new type of image change; the
synthetic regression problem tests a different data and loss family. Their
results remain exploratory because these task families were selected for this
PR. The study is CPU-only and does not test hardware throughput on a GPU.
