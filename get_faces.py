import kaolin as kal
import torch

with open("./face_numbers", "w") as myfile:
    mesh = kal.rep.TriangleMesh.from_obj("cow.obj")
    myfile.write("{}\n".format(mesh.faces.shape[0]))
    mesh = kal.rep.TriangleMesh.from_obj("brain.obj")
    myfile.write("{}\n".format(mesh.faces.shape[0]))
    mesh = kal.rep.TriangleMesh.from_obj("bunny.obj")
    myfile.write("{}\n".format(mesh.faces.shape[0]))
    mesh = kal.rep.TriangleMesh.from_obj("horse.obj")
    myfile.write("{}\n".format(mesh.faces.shape[0]))
    mesh = kal.rep.TriangleMesh.from_obj("maxplanck.obj")
    myfile.write("{}\n".format(mesh.faces.shape[0]))
