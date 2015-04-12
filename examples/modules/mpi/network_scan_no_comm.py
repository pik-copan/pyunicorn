import numpy
from pyunicorn import Network, mpi

offset = 10
n_max = 1000
s = 0
n = mpi.rank + offset
while n <= n_max + offset:
    s += Network.BarabasiAlbert(n_nodes=n).global_clustering()
    n += mpi.size

numpy.save("s"+str(mpi.rank), s)
