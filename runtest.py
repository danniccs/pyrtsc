import curvature
import dcurv
import pointareas
import normals_lib
import kaolin as kal
import numpy as np
import sys
import torch
from time import process_time
import pickle
import os.path

mypath = "./k1speeds_chol"

try:
    if os.path.getsize(mypath) > 0:
        myfile = open(mypath, "rb")
        speedDict = pickle.load(myfile)
        myfile.close()
    else:
        raise(OSError)
except OSError as e:
    speedDict = {'cow': [], 'brain': [], 'bunny': [], 'horse': [], 'maxplanck': []}

for k,v in speedDict.items():
    mesh = kal.rep.TriangleMesh.from_obj(k+".obj")
    t1 = process_time()
    k1, k2, e1, e2 = curvature.compute_curvatures(mesh, method="cholesky")
    t2 = process_time()
    v.append(t2 - t1)
    print("Finished processing {}.".format(k))

myfile = open(mypath, "wb")
pickle.dump(speedDict, myfile)
myfile.close()
