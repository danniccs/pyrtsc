import curvature
import dcurv
import pointareas
import kaolin as kal
import sys
import torch

path = str(sys.argv[1])
mesh = kal.rep.TriangleMesh.from_obj(path)
dcurv.compute_dcurvs(mesh)
