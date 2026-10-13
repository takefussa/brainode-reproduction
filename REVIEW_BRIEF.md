# Independent implementation review

## Objective

Assess whether the implementation follows the stated BrainODE reproduction protocol, identify defects affecting scientific results or reproducibility, and distinguish implementation supplements from unsupported claims. Start with README, SPEC, and the active LV config, then trace the referenced functions. Focus on the active `pipelines/lv_v1` snapshot as well as the completed hippocampal implementation at the repository root.

## Priority review areas

1. Patient-level leakage; diagnosis/age selection; train-only PCA, grid and time fitting; preservation of repeated/single scans; exposure of future diagnosis or test endpoints.
2. Algorithm 1 forward/backward time pairs, loss reductions, patient and short-batch weighting, condition handling, vertex-space scale augmentation and PCA mean offset.
3. Eq.(4) pseudo shape construction and fractional labels; separate endpoint trajectories; observed/pseudo mixing; CNN supervision; Algorithm 2 c updates and matched-solver comparisons.
4. Segmentation labels, NIfTI affine and unknown units, fixed-topology template axes, initial volume scaling, rigid ICP, longitudinal mesh initialization, native-grid Dice and self-intersection checks.
5. Voxel center occupancy, physical origin/pitch and pitch cubed volumes, grid bounds, three/seven-ray consistency, deterministic pooling, and microbatch gradient accumulation.
6. Evaluation against pre-PCA observed meshes, patient macro versus target micro means, static controls, forecast horizon, PCA reconstruction error, and paper comparisons with different cohorts/metrics.
7. Stage locks, partial artifacts, completion hashes, optimizer/scheduler/RNG checkpoints, real prefix/epoch resumption, two-GPU mapping, failure handling, and server-specific paths.

## Evidence and report format

For each confirmed issue, provide severity, exact file and line, trigger, consequence, evidence or a minimal reproducer, and a proposed fix. Separate confirmed defects, protocol deviations, acceptable documented supplements, and questions requiring unavailable data or publication details. State which tests were actually run. Do not infer absent clinical labels, repair geometry, remove observations, or rewrite code as part of the review. A passing synthetic test does not establish anatomical validity or paper-level accuracy.

## Relevant modules

- `lab_dataset.py`, `lab_shapes.py`: identity, chronology, splits and PCA.
- `mesh_input.py`, `mesh_initialization.py`, `mesh_fit.py`, `lv_mesh_fit.py`, `mesh_cohort.py`: observed geometry and prefix integrity.
- `model.py`, `paper_training.py`, `train_lab.py`: ODE model and observed training.
- `paper_cognition.py`, `cognition.py`, `deterministic_pool.py`, `train_continuous.py`: pseudo labels, CNN and progressive prediction.
- `evaluation.py`: metric definitions and aggregation.
- `pipelines/lv_v1/pipeline.py`, `cnn_microbatch.py`, `resume_utils.py`, `final_export.py`: active bilateral orchestration, accumulation, resumption and export.
- `audit/resume_lv_on_host.py`, `verify_lv_resume_inventory.py`, `matched_progressive_controls.py`: host migration and controls.

## Fixed limits

Original headers have unspecified units. CNN details, pseudo sampling/mixing, extra20 epochs and N=10 are supplements. The conversion supervision uses only two train patients and no held-out converters. Predicted/PCA surfaces can self-intersect. Per-patient ODE calls have singleton attention. Full medical image data, patient tables and checkpoints are external; claims requiring those inputs must remain unverified when reviewing this code-only repository.
