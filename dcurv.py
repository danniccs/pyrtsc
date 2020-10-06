import torch
import curvature as curv
from normals_lib import compute_simple_vertex_normals
from pointareas import compute_pointareas

pathdcurv = "/home/danniccs/torch_dcurv.txt"
pathdcurvold = "/home/danniccs/torch_dcurv_old.txt"
pathorigcurv= "/home/danniccs/torch_origcurv.txt"

# Igual que proj_curv pero para derivadas de curvatura
def proj_dcurv(old_u, old_v, old_dcurv, new_u, new_v):

    with open(pathdcurvold, "a") as myfile:
        myfile.write("old_u: ({}, {}, {}) old_v: ({}, {}, {})\n".format(old_u[0], old_u[1], old_u[2],
                                                                        old_v[0], old_v[1], old_v[2]
                                                                       ))

    r_new_u, r_new_v = curv.rot_coord_sys(new_u, new_v, torch.cross(old_u, old_v))

    u1 = r_new_u.dot(old_u)
    v1 = r_new_u.dot(old_v)
    u2 = r_new_v.dot(old_u)
    v2 = r_new_v.dot(old_v)

    new_dcurv = torch.zeros(4)
    new_dcurv[0] = (old_dcurv[0] * u1 * u1 * u1 +
                    old_dcurv[1] * 3 * u1 * u1 * v1 +
                    old_dcurv[2] * 3 * u1 * v1 * v1 +
                    old_dcurv[3] * v1 * v1 * v1)
    new_dcurv[1] = (old_dcurv[0] * u1 * u1 * u2 +
                    old_dcurv[1] * (u1 * u1 * v2 + 2 * u2 * u1 * v1) +
                    old_dcurv[2] * (u2 * v1 * v1 + 2 * u1 * v1 * v2) +
                    old_dcurv[3] * v1 * v1 * v2)
    new_dcurv[2] = (old_dcurv[0] * u1 * u2 * u2 +
                    old_dcurv[1] * (u2 * u2 * v1 + 2 * u1 * u2 * v2) +
                    old_dcurv[2] * (u1 * v2 * v2 + 2 * u2 * v2 * v1) +
                    old_dcurv[3] * v1 * v2 * v2)
    new_dcurv[3] = (old_dcurv[0] * u2 * u2 * u2 +
                    old_dcurv[1] * 3 * u2 * u2 * v2 +
                    old_dcurv[2] * 3 * u2 * v2 * v2 +
                    old_dcurv[3] * v2 * v2 * v2)

    with open(pathdcurv, "a") as myfile:
        myfile.write("dcurv: {}, {}, {}, {}\n".format(new_dcurv[0], new_dcurv[1], new_dcurv[2],
                                                    new_dcurv[3]))

    return new_dcurv

# Computa las derivadas de las curvaturas
def compute_dcurvs(mesh, method="lstsq"):
    with open(pathdcurv, "w") as myfile:
        pass
    with open(pathdcurvold, "w") as myfile:
        pass
    with open(pathorigcurv, "w") as myfile:
        pass

    verts = mesh.vertices
    faces = mesh.faces
    normals = compute_simple_vertex_normals(mesh)
    pointareas, cornerareas = compute_pointareas(mesh)
    k1, k2, pdir1, pdir2 = curv.compute_curvatures(mesh)

    with open(pathorigcurv, "w") as myfile:
        for i in range(0, pdir1.shape[0]):
            myfile.write("{}\n".format(pdir1[i]))

    dcurv = torch.zeros(verts.shape[0], 4, dtype=verts.dtype)

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

        # Proyecto de las coordenadas de cada vertice a las coordenadas de la cara
        # creadas en el paso anterior
        fcurv = torch.zeros(3, 3, dtype=torch.float32)
        for j in range (0, 3):
            curr_vert = faces[i,j]
            fcurv[j,0], fcurv[j,1], fcurv[j,2] = curv.proj_curv(pdir1[curr_vert], pdir2[curr_vert],
                                                                k1[curr_vert], 0, k2[curr_vert],
                                                                t, b)

        # Estimo la derivada de curvatura usando la variacion de la curvatura
        # en los bordes de la cara
        m = torch.zeros(4, dtype=torch.float32)
        w = torch.zeros(4, 4, dtype=torch.float32)
        for j in range(0, 3):
            dfcurv = fcurv[(j-1) % 3] - fcurv[(j+1) % 3]
            u = edges[j].dot(t)
            v = edges[j].dot(b)
            w[0,0] = w[0,0] + u**2
            w[0,1] = w[0,1] + u*v
            w[3,3] = w[3,3] + v**2

            m[0] = m[0] + u * dfcurv[0]
            m[1] = m[1] + v * dfcurv[0] + 2 * u * dfcurv[1]
            m[2] = m[2] + 2 * v * dfcurv[1] + u * dfcurv[2]
            m[3] = m[3] + v * dfcurv[2]

        w[1,1] = 2 * w[0,0] + w[3,3]
        w[1,2] = 2 * w[0,1]
        w[2,2] = w[0,0] + 2 * w[3,3]
        w[2,3] = w[0,1]

        # Encuentro la solucion de minimos cuadrados
        if method == "lstsq":
            m = torch.lstsq(m, w).solution
        elif method == "cholesky":
            chol = torch.cholesky(w)
            m = torch.cholesky_solve(m, chol)

        # Paso los valores a cada vertice
        for j in range(0, 3):
            curr_vert = faces[i,j]
            vert_dcurv = proj_dcurv(t, b, m, pdir1[curr_vert], pdir2[curr_vert])
            wt = cornerareas[i,j] / pointareas[curr_vert]
            dcurv[curr_vert] = dcurv[curr_vert] + wt * vert_dcurv

    return dcurv
