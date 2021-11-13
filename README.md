###### pyrtsc: Extraction of geometric properties from 3D models using PyTorch

### Overview

pyrtsc provides a set of functions to extract geometric properties from triangular meshes.
The algorithms are based on "Suggestive Contours for Conveying Shape" by Doug DeCarlo,
Adam Finkelstein, Szymon Rusinkiewicz, and Anthony Santella. The aim of this codebase is to
provide calculation of properties such as curvature and derivatives of curvature in PyTorch,
so it can be more easily integrated into research for automatic line drawing.

### Requirements

This library requires:
- [PyTorch](https://pytorch.org/) 1.4.0 or greater.
- [Numpy](https://numpy.org/) 1.19.2 or greater.

Also recommended, but not required:
- CUDA 10.1 or higher ([PyTorch](https://pytorch.org/) has instructions to install CUDA)
to speed up processing.
- A library such as [Kaolin](https://github.com/NVIDIAGameWorks/kaolin) to load 3D models.

To use the library simply download the repository and import pyrtsc into your program.

### Using pyrtsc

Using the library is quite simple:
- Load a triangular mesh.
- Import pyrtsc.
- Use curvature.compute_curvatures to calculate principal curvatures and their directions.
- Use dcurv.compute_dcurvs to calculate the derivative of curvature tensors.
- Use compute_perview to calculate the geometric properties used for Suggestive Contours.
- You can also use compute_simple_normals and compute_max_normals to compute per-vertex normals.

You can find a usage example in the visualization.ipynb jupyter notebook. For this visualization
you need all the libraries entioned in Requirements, plus:
- [Meshplot](https://github.com/skoch9/meshplot/) for visualization.
