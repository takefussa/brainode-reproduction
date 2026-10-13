import pickle
from trimesh import Trimesh

def compute_mesh_volume(coords, face):
    mesh = Trimesh(vertices=coords, faces = face)
    return mesh.volume
def train_test_dataset():
    train_d = pickle.load(open(rf'dataset\Shapes\train_hippo.pkl', 'rb'))
    test_d =  pickle.load(open(rf'dataset\Shapes\test_hippo.pkl', 'rb'))
    return train_d, test_d