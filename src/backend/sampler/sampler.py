import numpy as np
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
from shapely.geometry import LineString
from tqdm import tqdm_notebook as tqdm
import math


class MeshSampler:
    def __init__(self, mesh, check_faces=False):
        self.mesh = mesh
        self.sphere_points = trimesh.sample.sample_surface(mesh.bounding_sphere, count=1000)[0]*2
        self.correct_faces = {i:0 for i in range(len(mesh.faces))}
        self.ray_mesh = RayMeshIntersector(geometry=mesh)
        self.faces_centroids = self.mesh.triangles.mean(axis=1)
        self.correct_points = np.array([])
        if check_faces:
            self.compute_visible_faces()
        else:
            self.correct_faces = {i:1 for i in range(len(mesh.faces))}
    
    def visible_faces(self):
        return self.correct_faces
        
    def compute_visible_faces(self):
        for i, face in enumerate(tqdm(self.mesh.triangles)):
            for point in face:
                ray_directions = -(self.sphere_points - point)
                faces_hit = self.ray_mesh.intersects_first(self.sphere_points, ray_directions)
                if i in faces_hit:
                    self.correct_faces[i] = 1
        return self.correct_faces
    
    def sample_points(self, n_points=10000):
        points = trimesh.sample.sample_surface(self.mesh, count=n_points)
        correct_points = []
        normals_for_points = []
        for i, point in enumerate(tqdm(points[0])):
            if self.correct_faces[points[1][i]] == 1:
                correct_points += [point]
                normals_for_points += [self.mesh.face_normals[points[1][i]]]
        self.correct_points = np.array(correct_points)
        self.normals_for_points = np.array(normals_for_points)
        return self.correct_points
    
    def compute_sdf(self, sigma=0.0025):
        noise = np.random.normal(0, sigma, self.correct_points.shape)
        noisy_points = self.correct_points + noise
        sdf = self.mesh.nearest.signed_distance(noisy_points)
        correct_mesh_points = []
        correct_sdf = []
        correct_normals = []
        for i, distance in enumerate(sdf):
            if math.isnan(distance):
                continue
            else:
                correct_mesh_points += [noisy_points[i]]
                correct_sdf += [sdf[i]]
                correct_normals += [self.normals_for_points[i]]
        return np.array(correct_mesh_points), np.array(correct_sdf), np.array(correct_normals)