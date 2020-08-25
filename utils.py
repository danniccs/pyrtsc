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

def norm_sq(v, dim):
    return torch.sum(v**2, dim=dim)

def calculate_pointareas(mesh):
    verts = mesh.vertices
    faces = mesh.faces

    # Conseguir vectores de bordes
    e0 = verts[faces[:,2]] - verts[faces[:,1]]
    e1 = verts[faces[:,0]] - verts[faces[:,2]]
    e2 = verts[faces[:,1]] - verts[faces[:,0]]

    # Computar los pesos en cada esquina, basados en el area
    area = 0.5 * torch.norm(torch.cross(e0, e1), dim=1)
    l2 = torch.stack([norm_sq(e0, 1), norm_sq(e1, 1), norm_sq(e2, 1)], dim=1)

    # Pesos baricéntricos del circumcentro
    bcw = torch.stack([l2[:,0] * (l2[:,1] + l2[:,2] - l2[:,0]),
                       l2[:,1] * (l2[:,2] + l2[:,0] - l2[:,1]),
                       l2[:,2] * (l2[:,0] + l2[:,1] - l2[:,2])], dim=1)

    # Calculo las areas en las esquinas en base a los pesos baricéntricos
    pointareas = torch.zeros(verts.shape, dtype=verts.dtype)
    cornerareas = torch.zeros(faces.shape, dtype=verts.dtype)

    for i in range(0, faces.shape[0]):
            if bcw[i,0] <= 0.0:
                    cornerareas[i,1] = -0.25 * l2[i,2] * area[i] / torch.dot(e0[i], e2[i])
                    cornerareas[i,2] = -0.25 * l2[i,1] * area[i] / torch.dot(e0[i], e1[i])
                    cornerareas[i,0] = area[i] - cornerareas[i,1] - cornerareas[i,2]

            elif bcw[i,1] <= 0.0:
                    cornerareas[i,2] = -0.25 * l2[i,0] * area[i] / torch.dot(e1[i], e0[i])
                    cornerareas[i,0] = -0.25 * l2[i,2] * area[i] / torch.dot(e1[i], e2[i])
                    cornerareas[i,1] = area[i] - cornerareas[i,2] - cornerareas[i,0]

            elif bcw[i,2] <= 0.0:
                    cornerareas[i,0] = -0.25 * l2[i,1] * area[i] / torch.dot(e2[i], e1[i])
                    cornerareas[i,1] = -0.25 * l2[i,0] * area[i] / torch.dot(e2[i], e0[i])
                    cornerareas[i,2] = area[i] - cornerareas[i,0] - cornerareas[i,1]

            else:
                    scale = 0.5 * area[i] / (bcw[i,0] + bcw[i,1] + bcw[i,2])
                    cornerareas[i,0] = scale * (bcw[i,1] + bcw[i,2])
                    cornerareas[i,1] = scale * (bcw[i,2] + bcw[i,0])
                    cornerareas[i,2] = scale * (bcw[i,0] + bcw[i,1])

            pointareas[faces[i,0]] += cornerareas[i,0]
            pointareas[faces[i,1]] += cornerareas[i,1]
            pointareas[faces[i,2]] += cornerareas[i,2]

    return pointareas
