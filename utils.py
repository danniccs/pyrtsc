import torch
import kaolin as kal
import numpy as np
import meshplot as mp

def compute_face_areas(mesh):
    # Calculo las areas de cada triangulo, usando el hecho de que la normal del producto cruz
    # da el area de un paralelograma definido por los vectores. La mitad de eso da el area del
    # triangulo.
    faces = mesh.faces
    verts = mesh.vertices
    cross_products = torch.cross(verts[faces[:,1]] - verts[faces[:,0]],
                                 verts[faces[:,2]] - verts[faces[:,0]])
    face_areas = 0.5 * torch.norm(cross_products, dim = 1)
    return face_areas

def compute_vertex_normals(mesh):
    # Calculo las normales por vertice, calculando normales pesadas de cada cara y sumando ese
    # valor a las normales de los vertices correspondientes. Al final normalizo.
    faces = mesh.faces
    verts = mesh.vertices
    weighted_face_normals = torch.cross(verts[faces[:,1]] - verts[faces[:,0]],
                                        verts[faces[:,2]] - verts[faces[:,0]])
    vertex_normals = torch.zeros(verts.shape)

    for i in range(0, faces.size(0)):
            vertex_normals[faces[i][0]] += weighted_face_normals[i]
            vertex_normals[faces[i][1]] += weighted_face_normals[i]
            vertex_normals[faces[i][2]] += weighted_face_normals[i]

    vertex_normals = torch.nn.functional.normalize(vertex_normals)
    return vertex_normals

def visualize_normals(mesh, vertex_normals):
    color_normals = 0.5 + vertex_normals * 0.5
    mp.plot(mesh.vertices.numpy(), mesh.faces.numpy(), c=color_normals.numpy())
