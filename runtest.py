import curvature
import dcurv
import pointareas
import kaolin as kal
import numpy as np
import sys
import torch

mesh = kal.rep.TriangleMesh.from_obj("torus.obj")
curvature.compute_curvatures(mesh)
