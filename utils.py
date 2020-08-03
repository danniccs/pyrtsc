import torch
import kaolin as kal

def compute_face_areas(mesh):
    faces = mesh.faces
    verts = mesh.vertices
    cross_products = torch.cross(verts[faces[:,1]] - verts[faces[:,0]],
                                 verts[faces[:,2]] - verts[faces[:,0]])
    face_areas = 0.5 * torch.norm(cross_products, dim = 1)

def compute_vertex_normals(mesh):
    faces = mesh.faces
    verts = mesh.vertices
    weighted_face_normals = torch.cross(verts[faces[:,0]] - verts[faces[:,1]],
                                        verts[faces[:,1]] - verts[faces[:,2]])
    vertex_normals = torch.zeros(verts.shape)

    for i in range(0, faces.size(0)):
            vertex_normals[faces[i][0]] += weighted_face_normals[i]
            vertex_normals[faces[i][1]] += weighted_face_normals[i]
            vertex_normals[faces[i][2]] += weighted_face_normals[i]

    vertex_normals = torch.nn.functional.normalize(vertex_normals)
