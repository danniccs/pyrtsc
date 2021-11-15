import torch
import numpy as np


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

"""
Calculate the areas of each triangle using the cross product of the edges.
"""
def compute_face_areas(mesh):
    faces = mesh.faces.to(device=device)
    verts = mesh.vertices.to(device=device)

    crossProducts = torch.cross(verts[faces[:,1]] - verts[faces[:,0]],
                                verts[faces[:,2]] - verts[faces[:,0]])
    faceAreas = 0.5 * torch.norm(crossProducts, dim = 1)

    return face_areas

"""
Calculate the normals at each vertex using the normal of each triangle
and weighting using the area of the triangles.
"""
def compute_simple_normals(mesh):
    faces = mesh.faces.to(device=device)
    verts = mesh.vertices.to(device=device)

    vertexNormals = torch.zeros(verts.shape).to(device=device)

    weightedFaceNormals = torch.cross(verts[faces[:,1]] - verts[faces[:,0]],
                                      verts[faces[:,2]] - verts[faces[:,0]])

    for i in range(0, faces.size(0)):
            vertexNormals[faces[i][0]] = vertexNormals[faces[i][0]] + weightedFaceNormals[i]
            vertexNormals[faces[i][1]] = vertexNormals[faces[i][1]] + weightedFaceNormals[i]
            vertexNormals[faces[i][2]] = vertexNormals[faces[i][2]] + weightedFaceNormals[i]

    vertexNormals = torch.nn.functional.normalize(vertexNormals)

    return vertexNormals

"""
Calculate the normals at each vertex using the algorithm described in
Nelson Max, Weights for computing vertex normals from facet normals.
Journal of graphics tools, 1999.
"""
def compute_max_normals(mesh):
    faces = mesh.faces.to(device=device)
    verts = mesh.vertices.to(device=device)

    vertexNormals = torch.zeros(verts.shape).to(device=device)
    
    a = verts[faces[:,0]] - verts[faces[:,1]]
    b = verts[faces[:,1]] - verts[faces[:,2]]
    c = verts[faces[:,2]] - verts[faces[:,0]]
    weightedFaceNormals = torch.cross(a, b)
    l2a = torch.norm(a, dim=1)**2
    l2b = torch.norm(b, dim=1)**2
    l2c = torch.norm(c, dim=1)**2
    
    for i in range(0, faces.size(0)):
            vertexNormals[faces[i][0]] += weightedFaceNormals[i] * (1 / (l2a[i] * l2c[i]))
            vertexNormals[faces[i][1]] += weightedFaceNormals[i] * (1 / (l2b[i] * l2a[i]))
            vertexNormals[faces[i][2]] += weightedFaceNormals[i] * (1 / (l2c[i] * l2b[i]))
    
    vertexNormals = torch.nn.functional.normalize(vertexNormals)

    return vertexNormals
