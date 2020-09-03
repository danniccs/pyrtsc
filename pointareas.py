import torch
import meshplot as mp
import numpy as np
from utils import norm_sq

def visualize_pointareas(mesh, areas):
    max_area = torch.max(areas)
    red_areas = areas / max_area
    color_areas = torch.zeros(mesh.vertices.shape, dtype=mesh.vertices.dtype)
    color_areas[:,0] = red_areas
    mp.plot(mesh.vertices.numpy(), mesh.faces.numpy(), c=color_areas.numpy())

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
    pointareas = torch.zeros(verts.shape[0], dtype=verts.dtype)
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
