import numpy as np
import torch
import math

NUMSAMPLES = 12

def pol2cart(rho, phi):
    x = rho * math.cos(phi)
    y = rho * math.sin(phi)
    return x, y

def main():
    myfile = open("view_coordinates.pt", "w")
    myfile.close()
    coords = torch.zeros(NUMSAMPLES, 3)
    polcoords = np.zeros(NUMSAMPLES)
    for n in range(0, NUMSAMPLES):
        polcoords[n] = (2.0/NUMSAMPLES) * math.pi * n
        coords[n,0], coords[n,2] = pol2cart(2.0, polcoords[n])
    torch.save(coords, "view_coordinates.pt")

if __name__ == "__main__":
    main()
