from pyunicorn import Network, mpi


def master():
    net = Network.BarabasiAlbert(n_nodes=1000, n_links_each=10)
    print(net.newman_betweenness())
    mpi.info()

mpi.run()
