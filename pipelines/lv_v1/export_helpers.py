"""Coordinate-preserving display helpers copied from audited hippocampal exporter."""
from pathlib import Path
import hashlib
import numpy as np
import nibabel as nib
import trimesh

def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def write_mesh(prefix, vertices, faces):
    """Double precision RAS coordinates, original vertex/face ordering."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Invalid mesh')
    vtk, obj = prefix.with_suffix('.vtk'), prefix.with_suffix('.obj')
    with vtk.open('w') as file:
        file.write('# vtk DataFile Version 3.0\nBrainODE review. SPACE=RAS\nASCII\nDATASET POLYDATA\n')
        file.write(f'POINTS {len(vertices)} double\n')
        for vertex in vertices:
            file.write(' '.join(format(float(x), '.17g') for x in vertex) + '\n')
        file.write(f'POLYGONS {len(faces)} {4 * len(faces)}\n')
        for face in faces:
            file.write('3 ' + ' '.join(str(int(x)) for x in face) + '\n')
    with obj.open('w') as file:
        file.write('# BrainODE review. SPACE=RAS\n')
        for vertex in vertices:
            file.write('v ' + ' '.join(format(float(x), '.17g') for x in vertex) + '\n')
        for face in faces:
            file.write('f ' + ' '.join(str(int(x) + 1) for x in face) + '\n')
    restored = trimesh.load_mesh(obj, process=False)
    np.testing.assert_array_equal(restored.faces, faces)
    np.testing.assert_allclose(restored.vertices, vertices, atol=1e-12, rtol=0)
    # Read the primary VTK file back independently of the OBJ importer.
    lines = vtk.read_text().splitlines()
    assert 'SPACE=RAS' in lines[1] and lines[3] == 'DATASET POLYDATA'
    assert lines[4] == f'POINTS {len(vertices)} double'
    loaded_vertices = np.array([[float(x) for x in line.split()]
                               for line in lines[5:5 + len(vertices)]])
    assert lines[5 + len(vertices)] == f'POLYGONS {len(faces)} {4 * len(faces)}'
    loaded_faces = np.array([[int(x) for x in line.split()[1:]]
                            for line in lines[6 + len(vertices):]])
    np.testing.assert_array_equal(loaded_vertices, vertices)
    np.testing.assert_array_equal(loaded_faces, faces)
    return vtk, obj

def voxel_boundary(mask, affine):
    """Exact external faces of foreground voxel cells; no smoothing."""
    # Corner offsets are integers in a grid twice as fine as voxel indices.
    directions = [
        ((-1, 0, 0), [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)]),
        ((1, 0, 0), [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)]),
        ((0, -1, 0), [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)]),
        ((0, 1, 0), [(-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1)]),
        ((0, 0, -1), [(-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1)]),
        ((0, 0, 1), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]),
    ]
    corners, corner_indices, faces = [], {}, []
    shape = np.array(mask.shape)
    for center in np.argwhere(mask):
        for direction, offsets in directions:
            neighbor = center + direction
            if np.all(neighbor >= 0) and np.all(neighbor < shape) and mask[tuple(neighbor)]:
                continue
            quad = []
            for offset in offsets:
                corner = tuple(2 * center + offset)
                if corner not in corner_indices:
                    corner_indices[corner] = len(corners)
                    corners.append(corner)
                quad.append(corner_indices[corner])
            faces.extend([(quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])])
    vertices = nib.affines.apply_affine(affine, np.array(corners, dtype=np.float64) / 2)
    faces = np.array(faces, dtype=np.int64)
    if np.linalg.det(affine[:3, :3]) < 0:
        faces = faces[:, ::-1]
    volume = trimesh.Trimesh(vertices, faces, process=False).volume
    expected = int(mask.sum()) * abs(np.linalg.det(affine[:3, :3]))
    np.testing.assert_allclose(volume, expected, rtol=1e-10, atol=1e-8)
    return vertices, faces
