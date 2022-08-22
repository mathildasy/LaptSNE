import numpy as np
import dask.array as da

L = np.random.random((3000,3000))
L_da = da.from_array(L)
u, s, v = da.linalg.svd_compressed(L_da, k=10, compute=True)

u2 = u.compute()

print(u2.shape, type(u2))