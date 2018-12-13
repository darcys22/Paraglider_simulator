import numpy as np
import glob
import os

# numpy_vars = {}
numpy_vars = []
for np_name in glob.glob('./games/*.np[yz]'):
        # numpy_vars.append(np.load(np_name))
        base=os.path.basename(np_name)
        numpy_vars.append(float(os.path.splitext(base)[0]))

numpy_vars = sorted(numpy_vars)
for x in numpy_vars[3000:]:
    os.remove("./games/{}.npy".format(x))

print(numpy_vars[-1])
