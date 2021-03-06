import torch
import math
import curvature as curv
from normals_lib import compute_simple_normals
from pointareas import compute_pointareas
from perview import VIEWPOS


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

"""
Same as proj_curv but for derivatives of curvature
"""
def proj_dcurv(old_u, old_v, old_dcurv, new_u, new_v):
    r_new_u, r_new_v = curv.rot_coord_sys(new_u, new_v, torch.cross(old_u, old_v, dim=1))

    u1 = (r_new_u * old_u).sum(dim=1)
    v1 = (r_new_u * old_v).sum(dim=1)
    u2 = (r_new_v * old_u).sum(dim=1)
    v2 = (r_new_v * old_v).sum(dim=1)

    new_dcurv = torch.zeros(old_dcurv.shape).to(device=device)
    new_dcurv[:,0] = (old_dcurv[:,0] * u1 * u1 * u1 +
                      old_dcurv[:,1] * 3 * u1 * u1 * v1 +
                      old_dcurv[:,2] * 3 * u1 * v1 * v1 +
                      old_dcurv[:,3] * v1 * v1 * v1)
    new_dcurv[:,1] = (old_dcurv[:,0] * u1 * u1 * u2 +
                      old_dcurv[:,1] * (u1 * u1 * v2 + 2 * u2 * u1 * v1) +
                      old_dcurv[:,2] * (u2 * v1 * v1 + 2 * u1 * v1 * v2) +
                      old_dcurv[:,3] * v1 * v1 * v2)
    new_dcurv[:,2] = (old_dcurv[:,0] * u1 * u2 * u2 +
                      old_dcurv[:,1] * (u2 * u2 * v1 + 2 * u1 * u2 * v2) +
                      old_dcurv[:,2] * (u1 * v2 * v2 + 2 * u2 * v2 * v1) +
                      old_dcurv[:,3] * v1 * v2 * v2)
    new_dcurv[:,3] = (old_dcurv[:,0] * u2 * u2 * u2 +
                      old_dcurv[:,1] * 3 * u2 * u2 * v2 +
                      old_dcurv[:,2] * 3 * u2 * v2 * v2 +
                      old_dcurv[:,3] * v2 * v2 * v2)

    return new_dcurv

"""
Compute the derivatives of curvature (C) at each vertex
"""
def compute_dcurvs(mesh, method="cholesky", normals=None, pointareas=None,
                   cornerareas=None, curvs=None):

    verts = mesh.vertices.to(device=device)
    faces = mesh.faces.to(device=device)
    if normals == None:
        normals = compute_simple_normals(mesh)

    if pointareas == None or cornerareas == None:
        pointareas, cornerareas = compute_pointareas(mesh)

    if curvs == None:
        k1, k2, pdir1, pdir2 = curv.compute_curvatures(mesh)
    else:
        k1, k2, pdir1, pdir2 = curvs

    dcurv = torch.zeros(verts.shape[0], 4, dtype=verts.dtype).to(device=device)

    edges = torch.zeros(list(faces.shape) + [3], dtype=verts.dtype).to(device=device)
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

    fcurv = torch.zeros(faces.shape[0], 3, 3, dtype=torch.float32).to(device=device)
    for j in range(0, 3):
        vj = faces[:,j]
        fcurv[:,j,0], fcurv[:,j,1], fcurv[:,j,2] = curv.proj_curv(pdir1[vj], pdir2[vj],
                                                                  k1[vj], 0, k2[vj], t, b)

    # Estimate C using the variation of curvature across the edges of the triangle
    m = torch.zeros(faces.shape[0], 4, 1, dtype=torch.float32).to(device=device)
    w = torch.zeros(faces.shape[0], 4, 4, dtype=torch.float32).to(device=device)
    for j in range(0, 3):
        dfcurv = fcurv[:,(j-1) % 3] - fcurv[:,(j+1) % 3]
        u = (edges[:,j] * t).sum(dim=1)
        v = (edges[:,j] * b).sum(dim=1)
        w[:,0,0] = w[:,0,0] + u**2
        w[:,0,1] = w[:,0,1] + u*v
        w[:,3,3] = w[:,3,3] + v**2

        m[:,0] += (u * dfcurv[:,0]).view(m.shape[0], 1)
        m[:,1] += (v * dfcurv[:,0] + 2 * u * dfcurv[:,1]).view(m.shape[0], 1)
        m[:,2] += (2 * v * dfcurv[:,1] + u * dfcurv[:,2]).view(m.shape[0], 1)
        m[:,3] += (v * dfcurv[:,2]).view(m.shape[0], 1)

    w[:,1,1] = 2 * w[:,0,0] + w[:,3,3]
    w[:,1,2] = 2 * w[:,0,1]
    w[:,2,2] = w[:,0,0] + 2 * w[:,3,3]
    w[:,2,3] = w[:,0,1]

    # Use least squares to approximate a solution
    for i in range(0, faces.shape[0]):
        if method == "lstsq":
            m[i] = torch.lstsq(m[i], w[i]).solution
        elif method == "cholesky":
            chol = torch.cholesky(w[i])
            m[i] = torch.cholesky_solve(m[i], chol)

    # Propagate the values to the vertices
    for j in range(0, 3):
        vj = faces[:,j]
        vert_dcurv = proj_dcurv(t, b, m.view(m.shape[0], m.shape[1]), pdir1[vj], pdir2[vj])
        wt = cornerareas[:,j] / pointareas[vj]
        for i in range(0, faces.shape[0]):
            dcurv[vj[i]] += wt[i] * vert_dcurv[i]

    return dcurv

"""
Calculate the directional derivative of radial curvature in the direction
of the camera.
"""
def compute_DwKr(mesh, normals=None, curvs=None, dcurv=None, viewPos=VIEWPOS):
    verts = mesh.vertices.to(device=device)
    faces = mesh.faces.to(device=device)

    if normals == None:
        normals = compute_simple_normals(mesh)

    if curvs == None:
        k1, k2, pdir1, pdir2 = curv.compute_curvatures(mesh, normals=normals)
    else:
        k1, k2, pdir1, pdir2 = curvs

    if dcurv == None:
        dcurv = compute_dcurvs(mesh, normals=normals, curvs=(k1,k2,pdir1,pdir2))

    DwKr = torch.zeros(verts.shape[0], dtype=torch.float32).to(device=device)
    viewDir = -verts + viewPos
    ndotv = (viewDir * normals).sum(dim=1)

    w = viewDir - normals * ndotv[:,None]

    # Just as when calculating C, use pdir1 and pdir2 as bases for a coordinate system.
    u = (w * pdir1).sum(dim=1)
    v = (w * pdir2).sum(dim=1)
    u2 = u**2
    v2 = v**2

    cosphi = u.clone() / w.norm(dim=1)
    cos2phi = cosphi**2
    sin2phi = 1.0 - cos2phi
    sin2phi[sin2phi < 0.0] = 0.0
    sinphi = torch.sqrt(sin2phi)
    kr = k1 * cos2phi + k2 * sin2phi

    # DwKr = C(w,w,w) + 2Kcot(theta)
    DwKr = ( u2 * (u*dcurv[:,0] + 3.0*v*dcurv[:,1])
           + v2 * (3.0*u*dcurv[:,2] + v*dcurv[:,3]))

    K = k1 * k2

    # cot = cos/sin, and sin(theta) = ||w|| and cos(theta) = n.v
    DwKr *= w.norm(dim=1)
    DwKr += 2.0 * K * ndotv

    return DwKr, kr
