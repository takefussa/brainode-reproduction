# Bilateral lateral ventricle pipeline

ADNI1 full eligible cohort, separate left/right models, immutable patient split and train-fitted age/time scale. Original images, third-party sources, hippocampal experiments, and historical QC snapshots remain read-only.

The pipeline performs both cavity mesh fits, native-grid QC, train-only PCA150 and physical voxel grids, observed ODE100, train-only Eq.(4) augmentation, continuous-c CNN30, augmented ODE20, Algorithm 2 N=10, matched-solver controls, score comparison, and portable 3D export. Fixed final epochs; no test-driven selection. All scans and quality flags are retained. Invalid geometry or missing data stops the chain and records an error; no subject is removed to make a stage pass.

LV cavity labels are 163–167 left and 168–172 right. Choroid, third and fourth ventricles are separate labels. LV-Net supplies the audited, mirrored fixed-topology template. Its joint-anatomy loss is not used. BrainODE A.1 seven-loss PointNet fitting is reused, with complete native-voxel Dice in bounded slabs. Each fitted observation initializes the next observation of that patient. All mask components and same-day scans are retained.

CNN layers, soft BCE, voxel grid, pseudo sampling, epoch allocation, and N are explicit supplements where the publication omits details. Each side's 128³ isotropic grid is fitted only to train PCA surfaces (including coordinate scale .98–1.02), padded by 25% extent plus 40 mm total, with pitch rounded up to .125 mm. Test does not define grid bounds. Out-of-grid surfaces stop the stage rather than being cropped or rescaled. Occupancy volume uses pitch³; 3/7-ray consistency is checked. Original header units are unknown; 1 numerical coordinate unit is assumed to be 1 mm.

Checkpoints include optimizer, scheduler, Python/NumPy/Torch/CUDA RNG states and, for CNN, the shuffle generator state. Each stage locks its outputs. Complete stages have hash markers; unfinished noncheckpoint stages are archived before retry. Verified mesh prefixes and training epochs resume. Three bounded attempts record a persistent failure for review.

plant-ai05 hardware supplement: CUDA sm_75 PyTorch3D extension is loaded from a separate runtime overlay; original sm_86 installation remains unchanged. CNN uses microbatch4 with effective batch16, accumulating BCE sum/N gradients before one AdamW update. Its layers have no cross-example normalization or operations. Forward/backward memory and gradient equivalence are checked on RTX 2080 Ti. Cross-device numerical equality is not claimed. Data, objective, optimizer update count and checkpoint allocation stay fixed.

`smoke_chain.py` exercises the full chain with clearly labelled synthetic LV coefficients in separate directories. Its scores are not clinical or reproduction results. Original mathematical and deterministic-pooling tests also apply to the unchanged numerical modules.

The server supervisor runs in a detached session. `status.py REPORT` prints progress. Restart uses the same frozen config and code snapshot. A server shutdown, GPU fault, or data/geometry failure still requires checking the recorded status; closing a client computer does not terminate the detached process.
