import torch
import kaolin as kal
import numpy as np
import meshplot as mp
from utils import compute_vertex_normals
from pointareas import compute_pointareas

# Funcion para rotar un sistema de coordenadas para que sea perpendicular a la normal dada
def rot_coord_sys(old_u, old_v, new_norm):
    new_u = old_u
    new_v = old_v
    old_norm = torch.cross(old_u, old_v)
    ndot = torch.dot(old_norm, new_norm)

    if ndot <= -1.0:
        new_u = -new_u
        new_v = -new_v
        return new_u, new_v

    # La componente de new_norm perpendicular a old_norm
    perp_old = new_norm - ndot * old_norm

    # La diferencia de las perpendiculares (de old_norm y new_norm), ya normalizada
    dperp = 1.0 / (1 + ndot) * (old_norm + new_norm)

    # Resta el componente en la perpendicular vieja, y agrega al componente en la nueva
    # perpendicular
    new_u -= dperp * (new_u.dot(perp_old))
    new_v -= dperp * (new_v.dot(perp_old))
    return new_u, new_v

# Funcion para reproyectar un tensor de curvatura de una base (old_u, old_v)
# a una base (new_u, new_v)
def proj_curv(old_u, old_v, old_ku, old_kuv, old_kv, new_u, new_v):
    r_new_u, r_new_v = rot_coord_sys(new_u, new_v, torch.cross(old_u, old_v))

    u1 = r_new_u.dot(old_u)
    v1 = r_new_u.dot(old_v)
    u2 = r_new_v.dot(old_u)
    v2 = r_new_v.dot(old_v)
    new_ku = old_ku * u1**2 + old_kuv * (2.0 * u1 * v1) + old_kv * v1**2
    new_kuv = old_ku * u1*u2 + old_kuv * (u1*v2 + u2*v1) + old_kv * v1*v2;
    new_kv  = old_ku * u2**2 + old_kuv * (2.0 * u2 * v2) + old_kv * v2**2;
    return new_ku, new_kuv, new_kv

# Dado un tensor de curvatura (el mapa fundamental II), esta funcion encuentra
# las direcciones de curvatura principales y las curvaturas principales.
# (Estas son equivalentes a los autovectores y autovalores de la matriz, respectivamente)
# Asegura que pdir1 y pdir2 son perpendiculares a la normal.
def diagonalize_curv(old_u, old_v, ku, kuv, kv, new_norm):
    r_old_u, r_old_v = rot_coord_sys(old_u, old_v, new_norm)

    c = 1
    s = 0
    tt = 0
    if kuv != 0.0:
        # Rotacion Jacobiana para diagonalizar
        h = 0.5 * (kv - ku) / kuv
        if h < 0.0:
            tt = 1.0 / (h - np.sqrt(1.0 + h**2))
        else:
            tt = 1.0 / (h + np.sqrt(1.0 + h**2))
        c = 1.0 / np.sqrt(1.0 + tt**2)
        s = tt * c

    k1 = ku - tt * kuv
    k2 = kv + tt * kuv

    if abs(k1) >= abs(k2):
        pdir1 = c * r_old_u - s * r_old_v
    else:
        k1, k2 = k2, k1
        pdir1 = s * r_old_u + c * r_old_v

    pdir2 = torch.cross(new_norm, pdir1)

    return pdir1, pdir2, k1, k2

def compute_curvatures(mesh):
    verts = mesh.vertices
    faces = mesh.faces
    normals = compute_vertex_normals(mesh)
    pointareas, cornerareas = compute_pointareas(mesh)

    pdir1 = torch.zeros(verts.shape, dtype=verts.dtype)
    pdir2 = torch.zeros(verts.shape, dtype=verts.dtype)
    curv1 = torch.zeros(verts.shape[0], dtype=verts.dtype)
    curv12 = torch.zeros(verts.shape[0], dtype=verts.dtype)
    curv2 = torch.zeros(verts.shape[0], dtype=verts.dtype)

    # Creo un sistema de coordenadas inicial por cada vertice
    for i in range(0, faces.shape[0]):
        pdir1[faces[i, 0]] = verts[faces[i, 1]] - verts[faces[i, 0]]
        pdir1[faces[i, 1]] = verts[faces[i, 2]] - verts[faces[i, 1]]
        pdir1[faces[i, 2]] = verts[faces[i, 0]] - verts[faces[i, 2]]

    pdir1 = torch.cross(pdir1, normals)
    pdir1 = torch.nn.functional.normalize(pdir1)
    pdir2 = torch.cross(normals, pdir1)

    # Computar curvatura por cara
    for i in range(0, faces.shape[0]):
        # Bordes
        edges = torch.stack([verts[faces[i,2]] - verts[faces[i,1]],
                             verts[faces[i,0]] - verts[faces[i,2]],
                             verts[faces[i,1]] - verts[faces[i,0]]], dim=0)

        # Uso el marco de Frenet por cada cara
        t = edges[0]
        t = torch.nn.functional.normalize(t, dim=0)
        n = torch.cross(edges[0], edges[1])
        b = torch.cross(n, t)
        b = torch.nn.functional.normalize(b, dim=0)

        # Estimo la curvatura basado en la variacion de las normales
        m = torch.zeros(3)
        w = torch.zeros(3,3)
        for j in range(0, 3):
            u = edges[j].dot(t)
            v = edges[j].dot(b)
            w[0,0] += u**2
            w[0,1] += u*v
            w[2,2] += v*v

            dn = normals[faces[i, (j-1) % 3]] - normals[faces[i, (j+1) % 3]]
            dnu = dn.dot(t)
            dnv = dn.dot(b)
            m[0] += dnu * u
            m[1] += dnu * v + dnv * u
            m[2] += dnv * v

        w[1,1] = w[0,0] + w[2,2]
        w[1,2] = w[0,1]

        # Encuentro la solucion de minimos cuadrados
        m = torch.lstsq(m, w).solution
        for j in range(0, 3):
            vj = faces[i, j].item()
            c1, c12, c2 = proj_curv(t, b, m[0].item(), m[1].item(), m[2].item(),
                                    pdir1[vj], pdir2[vj])
            wt = cornerareas[i, j].item() / pointareas[vj].item()
            curv1[vj] += wt * c1
            curv12[vj] += wt * c12
            curv2[vj] += wt * c2

    # Computo direcciones y curvaturas principales en cada vertice
    for i in range(0, verts.shape[0]):
        pdir1[i], pdir2[i], curv1[i], curv2[i] = diagonalize_curv(pdir1[i], pdir2[i],
                                                                  curv1[i], curv12[i], curv2[i],
                                                                  normals[i])

    return curv1, curv12, curv2, pdir1, pdir2

def visualize_curvatures(mesh, k1, k2):
    max_k1 = torch.max(k1)
    max_k2 = torch.max(k2)
    red_k1 = k1 / max_k1
    red_k2 = k2 / max_k2
    color_k1 = torch.zeros(mesh.vertices.shape, dtype=mesh.vertices.dtype)
    color_k1[:,0] = red_k1
    color_k2 = torch.zeros(mesh.vertices.shape, dtype=mesh.vertices.dtype)
    color_k2[:,0] = red_k2
    mp.plot(mesh.vertices.numpy(), mesh.faces.numpy(), c=color_k1.numpy())
    mp.plot(mesh.vertices.numpy(), mesh.faces.numpy(), c=color_k2.numpy())
