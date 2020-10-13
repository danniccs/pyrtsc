import curvature
import dcurv
import pointareas
import kaolin as kal
import sys
import torch

path = str(sys.argv[1])
mesh = kal.rep.TriangleMesh.from_obj(path)
k1, k2, pdir1, pdir2 = curvature.compute_curvatures(mesh)
with open("pdir1.txt", "w") as myfile:
    for i in range(0, pdir1.shape[0]):
        myfile.write("{}\n".format(pdir1[i]))
