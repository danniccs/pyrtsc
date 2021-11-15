import torch
import numpy as np
from normals_lib import compute_simple_normals
from pointareas import compute_pointareas


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

"""
Function to rotate a coordinate system so it is perpendicular to the given normal.
"""
def rot_coord_sys(old_u, old_v, new_norm):
    new_u = old_u.clone()
    new_v = old_v.clone()
    perp_old = torch.zeros(new_u.shape, dtype=new_u.dtype).to(device=device)
    dperp = torch.zeros(new_u.shape, dtype=new_u.dtype).to(device=device)
    old_norm = torch.cross(old_u, old_v, dim=1)
    ndot = (old_norm * new_norm).sum(dim=1)

    mask = ndot > -1.0
    new_u[ndot <= -1.0] = -new_u[ndot <= -1.0]
    new_v[ndot <= -1.0] = -new_v[ndot <= -1.0]

    # La componente de new_norm perpendicular a old_norm
    perp_old = new_norm - old_norm * ndot[:,None]
    # La diferencia de las perpendiculares (de old_norm y new_norm), ya normalizada
    dperp = (old_norm + new_norm) * torch.reciprocal((ndot + 1.0))[:,None]
    # Resta el componente en la perpendicular vieja, y agrega al componente en la nueva
    # perpendicular
    new_u[mask] = new_u[mask] - (dperp[mask] * (new_u[mask] * perp_old[mask]).sum(dim=1)[:,None])
    new_v[mask] = new_v[mask] - (dperp[mask] * (new_v[mask] * perp_old[mask]).sum(dim=1)[:,None])

    return new_u, new_v

"""
Reproject a curvature tensor from one base to a new one.
"""
def proj_curv(old_u, old_v, old_ku, old_kuv, old_kv, new_u, new_v):
    r_new_u, r_new_v = rot_coord_sys(new_u, new_v, torch.cross(old_u, old_v, dim=1))

    # Dot product between r_new_u[i] and old_u[i] for all i
    u1 = (r_new_u * old_u).sum(dim=1)
    v1 = (r_new_u * old_v).sum(dim=1)
    u2 = (r_new_v * old_u).sum(dim=1)
    v2 = (r_new_v * old_v).sum(dim=1)

    new_ku = old_ku * u1**2 + old_kuv * (2.0 * u1 * v1) + old_kv * v1**2
    new_kuv = old_ku * u1*u2 + old_kuv * (u1*v2 + u2*v1) + old_kv * v1*v2;
    new_kv = old_ku * u2**2 + old_kuv * (2.0 * u2 * v2) + old_kv * v2**2;

    return new_ku, new_kuv, new_kv

"""
Given the second fundamental form (II), this function calculates the principal
curvatures and their directions. These are equivalent to the eigenvalues and
eigenvectors (respectively) of the tensor.
"""
def diagonalize_curv(old_u, old_v, ku, kuv, kv, new_norm):
    r_old_u, r_old_v = rot_coord_sys(old_u, old_v, new_norm)

    c = torch.ones(kuv.shape[0], dtype=torch.float32).to(device=device)
    s = torch.zeros(kuv.shape[0], dtype=torch.float32).to(device=device)
    tt = torch.zeros(kuv.shape[0], dtype=torch.float32).to(device=device)
    h = torch.zeros(kuv.shape[0], dtype=torch.float32).to(device=device)
    
    # Jacobi rotation to diagonalize
    kuvmask = kuv != 0.0
    h[kuvmask] = 0.5 * (kv[kuvmask] - ku[kuvmask]) / kuv[kuvmask]
    hmask1 = kuvmask == (h < 0.0)
    hmask2 = kuvmask == (h >= 0.0)
    tt[hmask1] = torch.reciprocal((h[hmask1] - torch.sqrt(1.0 + h[hmask1]**2)))
    tt[hmask2] = torch.reciprocal((h[hmask2] + torch.sqrt(1.0 + h[hmask2]**2)))
    c[kuvmask] = torch.reciprocal(torch.sqrt(1.0 + tt[kuvmask]**2))
    s[kuvmask] = tt[kuvmask] * c[kuvmask]

    k1 = ku - tt * kuv
    k2 = kv + tt * kuv

    pdir1 = torch.zeros(old_u.shape, dtype=torch.float32).to(device=device)

    posabsmask = torch.abs(k1) >= torch.abs(k2)
    negabsmask = torch.logical_not(posabsmask)
    pdir1[posabsmask] = ( c[:,None][posabsmask] * r_old_u[posabsmask]
                        - s[:,None][posabsmask] * r_old_v[posabsmask])
    k1[negabsmask], k2[negabsmask] = k2[negabsmask], k1[negabsmask]
    pdir1[negabsmask] = ( s[:,None][negabsmask] * r_old_u[negabsmask]
                        + c[:,None][negabsmask] * r_old_v[negabsmask])

    pdir2 = torch.cross(new_norm, pdir1, dim=1)

    return pdir1, pdir2, k1, k2

def diagonalize_curv_alt(old_u, old_v, ku, kuv, kv, new_norm):
    r_old_u, r_old_v = rot_coord_sys(old_u, old_v, new_norm)

    k1 = torch.ones(kuv.shape[0], dtype=torch.float32).to(device=device)
    k2 = torch.zeros(kuv.shape[0], dtype=torch.float32).to(device=device)
    pdir1 = torch.zeros(old_u.shape, dtype=torch.float32).to(device=device)
    pdir2 = torch.zeros(old_u.shape, dtype=torch.float32).to(device=device)
    
    # Jacobi rotation to diagonalize
    for i in range(0, kuv.shape[0]):
        if kuv[i] != 0.0:
            h = 0.5 * (kv[i] - ku[i]) / kuv[i]
            if h < 0.0:
                tt = 1.0 / (h - torch.sqrt(1.0 + h*h))
            else:
                tt = 1.0 / (h + torch.sqrt(1.0 + h*h))
            c = 1.0 / torch.sqrt(1.0 + tt*tt)
            s = tt * c

        k1[i] = ku[i] - tt * kuv[i]
        k2[i] = kv[i] + tt * kuv[i]

        if (torch.abs(k1[i]) >= torch.abs(k2[i])):
            pdir1[i] = c*r_old_u[i] - s*r_old_v[i]
        else:
            temp = k1[i]
            k1[i] = k2[i]
            k2[i] = temp
            pdir1[i] = s*r_old_u[i] + c*r_old_v[i]

        pdir2[i] = torch.cross(new_norm[i], pdir1[i])

    return pdir1, pdir2, k1, k2

"""
Compute the principal curvatures and their directions on the mesh.
The method for least squares can be lstsq or cholesky.
"""
def compute_curvatures(mesh, method="lstsq", normals=None, pointareas=None,
                       cornerareas=None):

    verts = mesh.vertices.to(device=device)
    faces = mesh.faces.to(device=device)

    if normals == None:
        normals = compute_simple_normals(mesh)

    if pointareas == None or cornerareas == None:
        pointareas, cornerareas = compute_pointareas(mesh)

    pdir1 = torch.zeros(verts.shape, dtype=verts.dtype).to(device=device)
    pdir2 = torch.zeros(verts.shape, dtype=verts.dtype).to(device=device)
    curv1 = torch.zeros(verts.shape[0], dtype=verts.dtype).to(device=device)
    curv12 = torch.zeros(verts.shape[0], dtype=verts.dtype).to(device=device)
    curv2 = torch.zeros(verts.shape[0], dtype=verts.dtype).to(device=device)

    # Create an initial coordinate system for each vertex
    for i in range(faces.shape[0]):
        pdir1[faces[i,0]] = verts[faces[i,1]] - verts[faces[i,0]]
        pdir1[faces[i,1]] = verts[faces[i,2]] - verts[faces[i,1]]
        pdir1[faces[i,2]] = verts[faces[i,0]] - verts[faces[i,2]]

    pdir1 = torch.cross(pdir1, normals)
    pdir1 = torch.nn.functional.normalize(pdir1)
    pdir2 = torch.cross(normals, pdir1)

    edges = torch.zeros(list(faces.shape) + [3], dtype=verts.dtype).to(device=device)

    # Compute the curvature of each face
    for i in range(0, faces.shape[0]):
        # Bordes
        edges[i] = torch.stack([verts[faces[i,2]] - verts[faces[i,1]],
                                verts[faces[i,0]] - verts[faces[i,2]],
                                verts[faces[i,1]] - verts[faces[i,0]]], dim=0)

    # Use the Frenet frame at each face
    t = edges[:,0]
    t = torch.nn.functional.normalize(t, dim=1)
    n = torch.cross(edges[:,0], edges[:,1])
    b = torch.cross(n, t)
    b = torch.nn.functional.normalize(b, dim=1)

    m = torch.zeros(faces.shape[0], 3, 1, dtype=torch.float32).to(device=device)
    w = torch.zeros(faces.shape[0], 3, 3, dtype=torch.float32).to(device=device)

    # Estimate the curvature based on the variation of normals
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
        # Use least squares to approximate a solution
        if method == "lstsq":
            m[i] = torch.lstsq(m[i], w[i]).solution
        elif method == "cholesky":
            chol = torch.cholesky(w[i])
            m[i] = torch.cholesky_solve(m[i], chol)

    # Sum the computed values at each vertex
    for j in range(0, 3):
        vj = faces[:,j]
        c1, c12, c2 = proj_curv(t, b, m[:,0].flatten(), m[:,1].flatten(), m[:,2].flatten(),
                                pdir1[vj], pdir2[vj])

        wt = cornerareas[:,j] / pointareas[vj]
        # This next part must be done inside a for to avoid race conditions
        for i in range(0, faces.shape[0]):
            curv1[vj[i]] += wt[i] * c1[i]
            curv12[vj[i]] += wt[i] * c12[i]
            curv2[vj[i]] += wt[i] * c2[i]

    pdir1, pdir2, curv1, curv2 = diagonalize_curv(pdir1, pdir2, curv1, curv12, curv2, normals)

    return curv1, curv2, pdir1, pdir2
