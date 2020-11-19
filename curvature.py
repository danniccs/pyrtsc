import torch
import kaolin as kal
import numpy as np
import meshplot as mp
from normals_lib import compute_simple_vertex_normals
from pointareas import compute_pointareas


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Funcion para rotar un sistema de coordenadas para que sea perpendicular a la normal dada
def rot_coord_sys(old_u, old_v, new_norm):
    new_u = old_u.clone()
    new_v = old_v.clone()
    perp_old = torch.zeros(new_u.shape, dtype=new_u.dtype).to(device=device)
    dperp = torch.zeros(new_u.shape, dtype=new_u.dtype).to(device=device)
    old_norm = torch.cross(old_u, old_v, dim=1)
    ndot = (old_norm * new_norm).sum(dim=1)

    mask = ndot > 1.0
    new_u[ndot <= 1.0] = -new_u[ndot <= 1.0]
    new_v[ndot <= 1.0] = -new_v[ndot <= 1.0]

    # La componente de new_norm perpendicular a old_norm
    perp_old = new_norm - old_norm * ndot[:,None]
    # La diferencia de las perpendiculares (de old_norm y new_norm), ya normalizada
    dperp = torch.reciprocal((old_norm + new_norm) * (ndot + 1.0)[:,None])
    # Resta el componente en la perpendicular vieja, y agrega al componente en la nueva
    # perpendicular
    new_u[mask] = new_u[mask] - (dperp[mask] * (new_u[mask] * perp_old[mask]).sum(dim=1)[:,None])
    new_v[mask] = new_v[mask] - (dperp[mask] * (new_v[mask] * perp_old[mask]).sum(dim=1)[:,None])

    return new_u, new_v

# Funcion para reproyectar un tensor de curvatura de una base (old_u, old_v)
# a una base (new_u, new_v)
def proj_curv(old_u, old_v, old_ku, old_kuv, old_kv, new_u, new_v):
    r_new_u, r_new_v = rot_coord_sys(new_u, new_v, torch.cross(old_u, old_v, dim=1))

    # Productos punto entre r_new_u[i] y old_u[i] para todo i
    u1 = (r_new_u * old_u).sum(dim=1)
    v1 = (r_new_u * old_v).sum(dim=1)
    u2 = (r_new_v * old_u).sum(dim=1)
    v2 = (r_new_v * old_v).sum(dim=1)

    new_ku = old_ku * u1**2 + old_kuv * (2.0 * u1 * v1) + old_kv * v1**2
    new_kuv = old_ku * u1*u2 + old_kuv * (u1*v2 + u2*v1) + old_kv * v1*v2;
    new_kv = old_ku * u2**2 + old_kuv * (2.0 * u2 * v2) + old_kv * v2**2;

    return new_ku, new_kuv, new_kv

# Dado un tensor de curvatura (el mapa fundamental II), esta funcion encuentra
# las direcciones de curvatura principales y las curvaturas principales.
# (Estas son equivalentes a los autovectores y autovalores de la matriz, respectivamente)
# Asegura que pdir1 y pdir2 son perpendiculares a la normal.
def diagonalize_curv(old_u, old_v, ku, kuv, kv, new_norm):
    r_old_u, r_old_v = rot_coord_sys(old_u, old_v, new_norm)

    c = torch.ones(kuv.shape[0], dtype=torch.float32).to(device=device)
    s = torch.zeros(kuv.shape[0], dtype=torch.float32).to(device=device)
    tt = torch.zeros(kuv.shape[0], dtype=torch.float32).to(device=device)
    for i in range(0, kuv.shape[0]):
        if kuv[i] != 0.0:
            # Rotacion Jacobiana para diagonalizar
            h = 0.5 * (kv[i] - ku[i]) / kuv[i]
            if h < 0.0:
                tt[i] = 1.0 / (h - torch.sqrt(1.0 + h**2))
            else:
                tt[i] = 1.0 / (h + torch.sqrt(1.0 + h**2))
            c[i] = 1.0 / torch.sqrt(1.0 + tt[i]**2)
            s[i] = tt[i] * c[i]

    k1 = ku - tt * kuv
    k2 = kv + tt * kuv

    pdir1 = torch.zeros(old_u.shape, dtype=torch.float32).to(device=device)

    for i in range(0, kuv.shape[0]):
        if abs(k1[i]) >= abs(k2[i]):
            pdir1[i] = c[i] * r_old_u[i] - s[i] * r_old_v[i]
        else:
            k1[i], k2[i] = k2[i], k1[i]
            pdir1[i] = s[i] * r_old_u[i] + c[i] * r_old_v[i]

    pdir2 = torch.cross(new_norm, pdir1, dim=1)

    return pdir1, pdir2, k1, k2

# Computa las curvaturas principales y sus direcciones sobre la malla
# method puede ser "lstsq" o "cholesky"
def compute_curvatures(mesh, method="lstsq", normals=None, pointareas=None,
                       cornerareas=None):

    verts = mesh.vertices.to(device=device)
    faces = mesh.faces.to(device=device)

    if normals == None:
        normals = compute_simple_vertex_normals(mesh)

    if pointareas == None or cornerareas == None:
        pointareas, cornerareas = compute_pointareas(mesh)

    pdir1 = torch.zeros(verts.shape, dtype=verts.dtype).to(device=device)
    pdir2 = torch.zeros(verts.shape, dtype=verts.dtype).to(device=device)
    curv1 = torch.zeros(verts.shape[0], dtype=verts.dtype).to(device=device)
    curv12 = torch.zeros(verts.shape[0], dtype=verts.dtype).to(device=device)
    curv2 = torch.zeros(verts.shape[0], dtype=verts.dtype).to(device=device)

    # Creo un sistema de coordenadas inicial por cada vertice
    for i in range(0, faces.shape[0]):
        pdir1[faces[i,0]] = verts[faces[i,1]] - verts[faces[i,0]]
        pdir1[faces[i,1]] = verts[faces[i,2]] - verts[faces[i,1]]
        pdir1[faces[i,2]] = verts[faces[i,0]] - verts[faces[i,2]]

    pdir1 = torch.cross(pdir1, normals)
    pdir1 = torch.nn.functional.normalize(pdir1)
    pdir2 = torch.cross(normals, pdir1)

    edges = torch.zeros(list(faces.shape) + [3], dtype=verts.dtype).to(device=device)

    # Computar curvatura por cara
    for i in range(0, faces.shape[0]):
        # Bordes
        edges[i] = torch.stack([verts[faces[i,2]] - verts[faces[i,1]],
                                verts[faces[i,0]] - verts[faces[i,2]],
                                verts[faces[i,1]] - verts[faces[i,0]]], dim=0)

    # Uso el marco de Frenet por cada cara
    t = edges[:,0]
    t = torch.nn.functional.normalize(t, dim=1)
    n = torch.cross(edges[:,0], edges[:,1])
    b = torch.cross(n, t)
    b = torch.nn.functional.normalize(b, dim=1)

    m = torch.zeros(faces.shape[0], 3, 1, dtype=torch.float32).to(device=device)
    w = torch.zeros(faces.shape[0], 3, 3, dtype=torch.float32).to(device=device)

    # Estimo la curvatura basado en la variacion de las normales
    for j in range(0, 3):
        u = (edges[:,j] * t).sum(dim=1)
        v = (edges[:,j] * b).sum(dim=1)
        w[:,0,0] += u*u
        w[:,0,1] += u*v
        w[:,2,2] += v*v

        dn = normals[faces[:, (j-1) % 3]] - normals[faces[:, (j+1) % 3]]
        dnu = (dn * t).sum(dim=1)
        dnv = (dn * b).sum(dim=1)
        m[:,0] += (dnu * u).view(m.shape[0], 1)
        m[:,1] += (dnu * v + dnv * u).view(m.shape[0], 1)
        m[:,2] += (dnv * v).view(m.shape[0], 1)

    w[:,1,1] = w[:,0,0] + w[:,2,2]
    w[:,1,2] = w[:,0,1]

    for i in range(0, faces.shape[0]):
        # Encuentro la solucion de minimos cuadrados
        # Agregué un if para seleccionar un método
        if method == "lstsq":
            m[i] = torch.lstsq(m[i], w[i]).solution
        elif method == "cholesky":
            chol = torch.cholesky(w[i])
            m[i] = torch.cholesky_solve(m[i], chol)

    # Sumar los valores computados en los vertices
    for j in range(0, 3):
        vj = faces[:,j]
        c1, c12, c2 = proj_curv(t, b, m[:,0].flatten(), m[:,1].flatten(), m[:,2].flatten(),
                                pdir1[vj], pdir2[vj])

        wt = cornerareas[:,j] / pointareas[vj]
        # Sumo a curvx[i] el producto de wt[i] y cx[i]
        # Esto debo hacerlo con un for para evitar errores de sincronización
        for i in range(0, faces.shape[0]):
            curv1[vj[i]] += wt[i] * c1[i]
            curv12[vj[i]] += wt[i] * c12[i]
            curv2[vj[i]] += wt[i] * c2[i]

    # Computo direcciones y curvaturas principales en cada vertice
    pdir1, pdir2, curv1, curv2 = diagonalize_curv(pdir1, pdir2, curv1, curv12, curv2, normals)

    return curv1, curv2, pdir1, pdir2
