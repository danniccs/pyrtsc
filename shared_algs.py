import torch
import kaolin as kal


SCTHRESH = 0.0
SHTHRESH = 0.0

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

VIEWPOS = torch.tensor([0.0, 0.0, -2.0], dtype=torch.float32).to(device=device)

#
#
# TODO: ADD FEATURE_SIZE CALCULATION
#
#

def compute_perview(mesh, normals=None, curvs=None, dcurv=None,
                    draw_apparent=False, view_coords=None):
    verts = mesh.vertices.to(device=device)
    faces = mesh.faces.to(device=device)

    if normals == None:
        normals = compute_simple_vertex_normals(mesh)

    if curvs == None:
        k1, k2, pdir1, pdir2 = curv.compute_curvatures(mesh)
    else:
        k1, k2, pdir1, pdir2 = curvs

    if dcurv == None:
        dcurv = compute_dcurvs(mesh, normals=normals, pointareas=pointareas,
                               cornerareas=cornerareas, curvs=(k1,k2,pdir1,pdir2))

    if view_coords == None:
        viewpos = VIEWPOS
    else:
        viewpos = view_coords

    scthresh = SCTHRESH
    shthresh = SHTHRESH

    ndotv = torch.zeros(verts.shape[0], dtype=torch.float32).to(device=device)
    kr = torch.zeros(verts.shape[0], dtype=torch.float32).to(device=device)
    viewdir = torch.zeros(verts.shape[0], 3, dtype=torch.float32).to(device=device)
    sctest_num = torch.zeros(verts.shape[0], dtype=torch.float32).to(device=device)
    sctest_den = torch.zeros(verts.shape[0], dtype=torch.float32).to(device=device)

    # Computar n.v
    viewdir = -verts + viewpos
    viewdir = torch.nn.functional.normalize(viewdir)
    ndotv = (viewdir * normals).sum(dim=1)

    u = (viewdir * pdir1).sum(dim=1)
    v = (viewdir * pdir2).sum(dim=1)
    u2 = u**2
    v2 = v**2

    # Esto en realidad es Kr * sin^2(theta)
    kr = k1 * u2 + k2 * v2

    # Uso DwKr * tan(theta) como umbral
    sctest_num = ( u2 * (u*dcurv[:,0] + 3.0*v*dcurv[:,1])
                 + v2 * (3.0*u*dcurv[:,2] + v*dcurv[:,3]))
    csc2theta = torch.reciprocal(u2 + v2)
    sctest_num *= csc2theta
    tr = (k2 - k1) * u * v * csc2theta
    sctest_num -= 2.0 * ndotv * tr**2

    sctest_num -= scthresh * ndotv

    return ndotv, kr, viewdir, sctest_num, ndotv

def find_zero_linear(val0, val1):
    return val0 / (val0 - val1)

def find_face_zeros_helper(mesh, v0, v1, v2, val, test_num, test_den):
    verts = mesh.vertices.to(device=device)

    w10 = find_zero_linear(val[v0], val[v1])
    w01 = 1.0 - w10
    w20 = find_zero_linear(val[v0], val[v2])
    w02 = 1.0 - w20

    p1 = w01 * verts[v0] + w10 * verts[v1]
    p2 = w02 * verts[v0] + w20 * verts[v2]

    test_num1 = test_num2 = 1.0
    test_den1 = test_den2 = 1.0
    z1 = 0.0
    z2 = 0.0
    valid1 = True

    test_num1 = w01 * test_num[v0] + w10 * test_num[v1]
    test_num2 = w02 * test_num[v0] + w20 * test_num[v2]
    if test_den != None:
        test_den1 = w01 * test_den[v0] + w10 * test_den[v1]
        test_den2 = w02 * test_den[v0] + w20 * test_den[v2]

    # El primer punto es valido sii num1/den1 > 0 (tienen el mismo signo)
    valid1 = ((test_num1 >= 0.0) == (test_den1 >= 0.0))
    if (test_num1 >= 0.0) != (test_num2 >= 0.0):
        z1 = test_num1 / (test_num1 - test_num2)
    if (test_den1 >= 0.0) != (test_den2 >= 0.0):
        z2 = test_den1 / (test_den1 - test_den2)
    # Ordeno los cruces
    if z1 == 0.0:
        z1 = z2
        z2 = 0.0
    elif z2 < z1:
        z1, z2 = z2, z1

    # Si el principio del segmento es invalido, y no hay cruces, todo el segmento
    # es invalido.
    if not valid1 or not z1 or not z2:
        return None, None

    return p1, p2

def find_face_zeros(mesh, v0, v1, v2, val, test_num, test_den, ndtov):
    # Testeo rapido en base a si las derivadas son negativas
    if test_den == None:
        if test_num[v0] <= 0.0 and test_num[v1] <= 0.0 and test_num[v2] <= 0.0:
            return None, None
    else:
        if (test_num[v0] <= 0.0 and test_den[v0] >= 0.0 and 
            test_num[v1] <= 0.0 and test_den[v1] >= 0.0 and
            test_num[v2] <= 0.0 and test_den[v2] >= 0.0):
            return None, None
        if (test_num[v0] >= 0.0 and test_den[v0] <= 0.0 and
            test_num[v1] >= 0.0 and test_den[v1] <= 0.0 and
            test_num[v2] >= 0.0 and test_den[v2] <= 0.0):
            return None, None

    p1 = p2 = None
    # Calculo cual valor tiene un signo distinto y encuentro los ceros
    if ((val[v0] < 0.0 and val[v1] >= 0.0 and val[v2] >= 0.0) or
        (val[v0] > 0.0 and val[v1] <= 0.0 and val[v2] <= 0.0)):
        p1,p2 = find_face_zeros_helper(mesh, v0, v1, v2, val, test_num, test_den)
    elif ((val[v1] < 0.0 and val[v2] >= 0.0 and val[v0] >= 0.0) or
             (val[v1] > 0.0 and val[v2] <= 0.0 and val[v0] <= 0.0)):
        p1,p2 = find_face_zeros_helper(mesh, v1, v2, v0, val, test_num, test_den)
    elif ((val[v2] < 0.0 and val[v0] >= 0.0 and val[v1] >= 0.0) or
             (val[v2] > 0.0 and val[v0] <= 0.0 and val[v1] <= 0.0)):
        p1,p2 = find_face_zeros_helper(mesh, v2, v0, v1, val, test_num, test_den)

    return p1, p2
