import os

for i in range(0, 10):
    exec(open('runtest.py').read())
    print("Finished run number {}".format(i+1))
