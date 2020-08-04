import torch
import kaolin as kal
import numpy as np
import ipyvolume as ipv
from PIL import Image

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
    weighted_face_normals = torch.cross(verts[faces[:,0]] - verts[faces[:,1]],
                                        verts[faces[:,1]] - verts[faces[:,2]])
    vertex_normals = torch.zeros(verts.shape)

    for i in range(0, faces.size(0)):
            vertex_normals[faces[i][0]] += weighted_face_normals[i]
            vertex_normals[faces[i][1]] += weighted_face_normals[i]
            vertex_normals[faces[i][2]] += weighted_face_normals[i]

    vertex_normals = torch.nn.functional.normalize(vertex_normals)
    return vertex_normals

def export_normals_as_texture(vertex_normals, image_path="normal_texture.png"):
    # Paso las normales al rango [0, 1], convirtiendolos a RGB
    color_normals = 0.5 + vertex_normals * 0.5
    # Convierto el tensor a un array de numpy para usar con PIL y con ipyvolume
    # Paso los colores a RGB, con valores entre 0 y 255 y expando una dimension para poder
    # utilizarlo como imágen.
    np_color_norms = np.uint8(color_normals.numpy() * 255)
    np_color_norms = np.expand_dims(np_color_norms, 0)
    image = Image.fromarray(np_color_norms, mode='RGB')
    # Guardo la imágen
    image.save(image_path)

def visualize_normals(mesh, normal_texture):
    # normal_texture debe ser una textura de num_vertices * 1. Construyo el mapa de coordenadas uv
    u = np.array([i/mesh.vertices.size(0) for i in range(0, mesh.vertices.size(0))])
    v = 0.5 * np.ones(mesh.vertices.size(0))
    # Convierto las coordenadas de vertices al formato esperado por ipyvolume
    x = mesh.vertices[:, 0].numpy()
    y = mesh.vertices[:, 1].numpy()
    z = mesh.vertices[:, 2].numpy()
    # Defino las caras en el formato esperado por ipyvolume
    faces = mesh.faces.numpy()
    # Dibujo el mesh con la textura
    ipv.figure()
    ipv_mesh = ipv.plot_trisurf(x, y, z, triangles=faces, u=u, v=v, texture=normal_texture)
    ipv.show()
