# BrainODE reproduction: hippocampus and lateral ventricles

BrainODE (NeurIPS 2025) reproduction using a laboratory's existing longitudinal parcellations. This repository contains source code, synthetic unit tests, fixed settings, and generic public/derived templates for independent review. Research images, patient tables, fitted patient meshes, PCA observations, voxel caches, model weights, and runtime logs are stored separately.

## Start the review here

1. [REVIEW_BRIEF.md](REVIEW_BRIEF.md): scope, review questions, and expected report format.
2. [SPEC.md](SPEC.md): hippocampal implementation, assumptions, and paper/code differences.
3. [pipelines/lv_v1/README.md](pipelines/lv_v1/README.md): bilateral lateral ventricle chain.
4. [docs/active_lv_config.json](docs/active_lv_config.json): exact active runtime configuration.
5. [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json): source hashes and active-snapshot correspondence.

The files under `pipelines/lv_v1/` are copied byte-for-byte from the immutable code snapshot used by the running bilateral LV experiment. The eligible patient CSV is intentionally stored separately. The source directory's historical name contains `plant05`; the current runtime uses plant-ai06, physical GPUs 0 and 1. CNN microbatch4 retains effective optimizer batch16. The native sm86 PyTorch3D extension is used on RTX A6000; compiled libraries are not uploaded.

Root-level reproduction modules implement mesh fitting, train-only PCA, observed BrainODE, Eq.(4) pseudo shapes, continuous-c CNN, and Algorithm 2 forecasting. Original upstream `model.py`, `dataset.py`, and `train.py` are retained to compare behavior. `audit/` contains preparation, quality checks, comparisons, exports, and restart helpers.

## Scientific settings

- ADNI1: 468 patients / 1,405 scans; CN/AD at scan level, with MCI scans excluded. No fixed 65–95 age cutoff.
- Patient split: train 389 / test 79; inherited merged-RID seed42 split.
- Time: `(age - 55.7288) / 37.2931`, fitted on train only; no test refit or clipping.
- Hippocampus: left label75, 1,490 vertices / 2,976 faces.
- LV: separate left labels163–167 and right labels168–172, 2,490 vertices / 4,976 faces per side.
- Mesh fitting: 5,000 PointNet updates per scan, fixed source within a scan, previous fitted observation as the next source. All components, repeats, and quality flags retained.
- PCA150: all train scans; no whitening.
- Initial ODE100 epochs; train-only pseudo samples; CNN30 epochs; augmented ODE20 epochs; Algorithm 2 N=10. Fixed final epochs, without test-based selection.
- Primary error: patient-macro corresponding-vertex Euclidean distance against observed meshes before PCA. Coordinate MAE against PCA targets is a separate metric.
- Original spatial units are unspecified; one numerical coordinate unit is interpreted as one mm.

CNN architecture/loss/grid, pseudo mixing and sampling, additional epochs, N, and initial template volume scaling are implementation supplements. They are documented rather than represented as fully recovered unpublished author settings.

## Checks and execution limits

The copied repository passed all 55 existing synthetic unit tests on CPU; 98 Python files passed syntax compilation. See [PUBLISH_VALIDATION.json](PUBLISH_VALIDATION.json). This does not validate full-dataset anatomical accuracy.

With the mesh environment and PyTorch3D installed, synthetic unit tests run with:

```bash
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
```

`environment.yml` is an upstream legacy file retained with its original non-UTF-8 bytes. Reproduction dependencies are specified by the requirement files.

`requirements-mesh.txt` records the server environment; PyTorch3D is built separately from pinned v0.7.9 source. `requirements-qc.txt` and `requirements-cognition.txt` document CPU checks. Full training requires the local research dataset and generated assets, which are not part of this repository. Many preparation and restart scripts still use the original server paths; path portability is a review topic. Config hashes referring to excluded assets describe the actual experiment, not bundled placeholders.

## Status and interpretation

Left hippocampal training and evaluation are complete. Latest-time error is 0.4258 assumed mm for the initial ODE and 0.4332 for the continuous-c final model, versus 0.6133 for a static initial mesh. The final additions did not improve the initial baseline. Residual self-intersections and the absence of held-out CN→AD converters limit interpretation. The bilateral LV experiment is still in progress; no LV forecasting accuracy is asserted here.

## Upstream sources

- [BrainODE paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/abdc8c031aa6c6917c3b593166e5e340-Paper-Conference.pdf).
- [BrainODE source](https://github.com/PWonjung/BrainODE/tree/a5e1a2ac2bcbfc1a0aa6cdafafcfdf101135b18d).
- [BrainSubcorticalStructure](https://github.com/PWonjung/BrainSubcorticalStructure/tree/d65ffe5da55e53e1431cc93f91e3b348497b8e5d): PointNet and loss source.
- [LV_Shape_Modeling](https://github.com/PWonjung/LV_Shape_Modeling/tree/6c9e5bbca8cb38f71ea5fe6f8c61258bd185ce11): generic LV template source.

Upstream attribution is preserved. No new blanket license is assigned to third-party files.
