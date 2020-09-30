# Igual que proj_curv pero para derivadas de curvatura
def proj_dcurv(old_u, old_v, old_dcurv, new_u, new_v):
    r_new_u, r_new_v = rot_coord_sys(new_u, new_v, torch.cross(old_u, old_v))

    u1 = r_new_u.dot(old_u)
    v1 = r_new_u.dot(old_v)
    u2 = r_new_v.dot(old_u)
    v2 = r_new_v.dot(old_v)

    new_dcurv = torch.zeros(4)
    new_dcurv[0] = (old_dcurv[0] * u1 * u1 * u1 +
                    old_dcurv[1] * 3.0 * u1 * u1 * v1 +
                    old_dcurv[2] * 3.0 * u1 * v1 * v1 +
                    old_dcurv[3] * v1 * v1 * v1)
    new_dcurv[1] = (old_dcurv[0] * u1 * u1 * u2 +
                    old_dcurv[1] * (u1 * u1 * v2 + 2.0 * u2 * u1 * v1) +
                    old_dcurv[2] * (u2 * v1 * v1 + 2.0 * u1 * v1 * v2) +
                    old_dcurv[3] * v1 * v1 * v2)
    new_dcurv[2] = (old_dcurv[0] * u1 * u2 * u2 +
                    old_dcurv[1] * (u2 * u2 * v1 + 2.0 * u1 * u2 * v2) +
                    old_dcurv[2] * (u1 * v2 * v2 + 2.0 * u2 * v2 * v1) +
                    old_dcurv[3] * v1 * v2 * v2)
    new_dcurv[3] = (old_dcurv[0] * u2 * u2 * u2 +
                    old_dcurv[1] * 3.0 * u2 * u2 * v2 +
                    old_dcurv[2] * 3.0 * u2 * v2 * v2 +
                    old_dcurv[3] * v2 * v2 * v2)

    return new_dcurv

# Computa las derivadas de las curvaturas
def compute_dcurvs(mesh):
    verts = mesh.vertices
    faces = mesh.faces
    normals = compute_vertex_normals(mesh)
    pointareas, cornerareas = compute_pointareas(mesh)
    k1, k2, pdir1, pdir2 = compute_curvatures(mesh)

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
        for j in range (0, 3):
            curr_vert = faces[i, j]
