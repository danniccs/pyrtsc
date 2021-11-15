import torch


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

VIEWPOS = torch.tensor([0.0, 0.0, -2], dtype=torch.float32).to(device=device)

"""
TODO: ADD FEATURE_SIZE CALCULATION
"""

def compute_perview(mesh, normals=None, curvs=None, dcurv=None,
                    scthresh=0.0, viewPos=VIEWPOS):
    verts = mesh.vertices.to(device=device)
    faces = mesh.faces.to(device=device)

    if normals == None:
        normals = compute_simple_normals(mesh)

    if curvs == None:
        k1, k2, pdir1, pdir2 = curv.compute_curvatures(mesh)
    else:
        k1, k2, pdir1, pdir2 = curvs

    if dcurv == None:
        dcurv = compute_dcurvs(mesh, normals=normals, pointareas=pointareas,
                               cornerareas=cornerareas, curvs=(k1,k2,pdir1,pdir2))

    ndotv = torch.zeros(verts.shape[0], dtype=torch.float32).to(device=device)
    kr = torch.zeros(verts.shape[0], dtype=torch.float32).to(device=device)
    viewdir = torch.zeros(verts.shape[0], 3, dtype=torch.float32).to(device=device)
    sctest = torch.zeros(verts.shape[0], dtype=torch.float32).to(device=device)

    # Compute n.v
    viewdir = -verts + viewPos
    viewdir = torch.nn.functional.normalize(viewdir)
    ndotv = (viewdir * normals).sum(dim=1)

    u = (viewdir * pdir1).sum(dim=1)
    v = (viewdir * pdir2).sum(dim=1)
    u2 = u**2
    v2 = v**2

    # This is actually Kr * sin^2(theta)
    kr = k1 * u2 + k2 * v2

    # We use DwKr*tan(theta) as threshold
    sctest = ( u2 * (u*dcurv[:,0] + 3.0*v*dcurv[:,1])
                 + v2 * (3.0*u*dcurv[:,2] + v*dcurv[:,3]))
    csc2theta = torch.reciprocal(u2 + v2)
    sctest *= csc2theta
    tr = (k2 - k1) * u * v * csc2theta
    sctest -= 2.0 * ndotv * tr**2

    sctest -= scthresh * ndotv

    return ndotv, kr, viewdir, sctestNum, ndotv
