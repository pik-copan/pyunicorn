from pyunicorn import Network, mpi


def do_one():
    net = Network.BarabasiAlbert(n_nodes=100, n_links_each=10)
    return net.global_clustering()


def master():
    n = 1000
    for i in range(0, n):
        mpi.submit_call("do_one", ())
    s = 0
    for i in range(0, n):
        s += mpi.get_next_result()
    print(s/n)
    mpi.info()

mpi.run()
