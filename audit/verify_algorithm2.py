"""Compare the batched evaluator with the published Algorithm 2 reference."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from cognition import CognitionEstimator3D, PCAShapeVoxelizer, RK4ShapeIntegrator
from deterministic_pool import DisjointMaxPool3D
from lab_shapes import ShapePCA
from model import ODEFuncWithAttention
from paper_cognition import progressive_prediction


def main():
    report = PROJECT/'artifacts/training_runs/adni1_continuous_v1'
    config = json.loads((report/'deterministic_pool_config.json').read_text())
    output = Path(config['output'])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    with np.load(Path(config['data_root'])/'observations.npz', allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in archive.files}
    with np.load(output/'evaluate/future_predictions.npz', allow_pickle=False) as archive:
        predictions = {k: archive[k].copy() for k in archive.files}
    pca = ShapePCA.load(Path(config['data_root'])/'pca.npz')
    ode = ODEFuncWithAttention(150, 1).to(device)
    ode.load_state_dict(torch.load(output/'ode/latest.pt', map_location=device, weights_only=False)['model']); ode.eval()
    cnn = CognitionEstimator3D().to(device)
    for i, layer in enumerate(cnn.features):
        if isinstance(layer, torch.nn.MaxPool3d):
            cnn.features[i] = DisjointMaxPool3D()
    cnn.load_state_dict(torch.load(output/'cnn/latest.pt', map_location=device, weights_only=False)['model']); cnn.eval()
    voxel = PCAShapeVoxelizer(pca, data['faces'], **config['grid'], occupancy_rays=3)
    indices = {str(uid): i for i, uid in enumerate(data['uids'])}
    # Deterministic first saved target, independent of its error or geometry.
    source, target = indices[str(predictions['source_uids'][0])], indices[str(predictions['target_uids'][0])]
    c_values = []

    def estimate(value):
        c = cnn.estimate_c(value); c_values.append(c); return c

    with torch.inference_mode():
        result = progressive_prediction(torch.tensor(data['coefficients'][source], dtype=torch.float32, device=device),
            float(data['times'][source]), float(data['times'][target]), steps=config['progressive_steps'],
            integrate=RK4ShapeIntegrator(ode), shape_to_voxels=voxel, estimate_c=estimate)
    coefficients = result.cpu().numpy()
    vertices = pca.inverse_transform(coefficients[None]).reshape(-1, 3)
    difference = np.linalg.norm(vertices-predictions['vertices'][0], axis=1)
    if difference.max() > 1e-4:
        raise ValueError('batched forecasting differs materially from Algorithm 2 reference')
    summary = dict(passed=True, target_uid=str(predictions['target_uids'][0]), source_uid=str(predictions['source_uids'][0]),
        pid=str(predictions['pids'][0]), steps=config['progressive_steps'], estimated_c=c_values,
        max_coefficient_difference=float(np.abs(coefficients-predictions['coefficients'][0]).max()),
        mean_vertex_difference_assumed_mm=float(difference.mean()), max_vertex_difference_assumed_mm=float(difference.max()),
        tolerance_max_vertex_assumed_mm=1e-4,
        scope='One deterministic real test trajectory plus analytic Algorithm 2 tests; not exhaustive trajectory equivalence.',
        future_diagnosis_supplied=False, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (report/'evaluate/reference_verification.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
