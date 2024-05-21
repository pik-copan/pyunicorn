# This file is part of pyunicorn.
# Copyright (C) 2008--2024 Jonathan F. Donges and pyunicorn authors
# URL: <https://www.pik-potsdam.de/members/donges/software-2/software>
# License: BSD (3-clause)
#
# Please acknowledge and cite the use of this software and its authors
# when results are used in publications or published elsewhere.
#
# You can use the following reference:
# J.F. Donges, J. Heitzig, B. Beronov, M. Wiedermann, J. Runge, Q.-Y. Feng,
# L. Tupikina, V. Stolbova, R.V. Donner, N. Marwan, H.A. Dijkstra,
# and J. Kurths, "Unified functional network and nonlinear time series analysis
# for complex systems science: The pyunicorn package"

"""
Module for parallelization using mpi4py.

Allows for easy parallelization in master/slaves mode with one master
submitting function or method calls to slaves.
Uses mpi4py if available, otherwise processes calls sequentially in one
process.

Examples:
=========

Save the following lines in ``demo_mpi.py`` and run::

    > mpirun -n 10 python demo_mpi.py

1. Use master/slaves parallelization with the Network class:

   .. literalinclude::
        ../../../../docs/source/examples/modules/mpi/network_large.py

2. Do a Monte Carlo simulation as master/slaves:

   .. literalinclude::
        ../../../../docs/source/examples/modules/mpi/network_mc.py

3. Do a parameter scan without communication with a master, and just save
   the results in files:

   .. literalinclude::
        ../../../../docs/source/examples/modules/mpi/network_scan_no_comm.py

"""

#
#  Imports
#

import sys
import time
import traceback

import numpy


# try to get the communicator object to see whether mpi is available:
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    """(mpi4py.MPI.Comm instance) MPI communicator."""
    available = True
    """(bool) indicates that slaves are available."""
except ImportError:
    available = False


class MPIException(Exception):
    def __init__(self, value):
        Exception.__init__()
        self.value = value

    def __str__(self):
        return repr(self.value)


# initialize:

if available:
    size = comm.size
    """(int) number of MPI nodes (master and slaves)."""
    rank = comm.rank
    """(int) rank of this MPI node (0 is the master)."""
    am_master = (rank == 0)
    """(bool) indicates that this MPI node is the master."""
    if size < 2:
        available = False
else:
    size = 1
    rank = 0
    am_master = True

am_slave = not am_master
"""(bool) indicates that this MPI node is a slave."""
n_slaves = size - 1
"""(int) no. of slaves available."""
start_time = time.time()
"""(float) starting time of this MPI node."""
stats = []
"""
(list of dictionaries)
stats[id] contains processing statistics for the last call with this id. Keys:

  - "id": id of the call
  - "rank": MPI node who processed the call
  - "this_time": wall time for processing the call
  - "time_over_est": quotient of actual over estimated wall time
  - "n_processed": no. of calls processed so far by that slave, including this
  - "total_time": total wall time until this call was finished
"""

# initialization on master with slaves:

if am_master:

    total_time_est = numpy.zeros(size)
    """
    (numpy array of ints)
    total_time_est[i] is the current estimate of the total time
    MPI slave i will work on already submitted calls.
    On slave i, only total_time_est[i] is available.
    """
    total_time_est[0] = numpy.inf
    queue = []
    """(list) ids of submitted calls"""
    assigned = {}
    """
    (dictionary)
    assigned[id] is the slave assigned to the call with that id.
    """
    slave_queue = [[] for i in range(0, size)]
    """
    (list of lists)
    slave_queue[i] contains the ids of calls assigned to slave i.
    """
    n_processed = numpy.zeros(size).astype("int")
    """
    (list of ints)
    n_processed[rank] is the total number of calls processed by MPI node rank.
    On slave i, only total_time[i] is available.
    """
    total_time = numpy.zeros(size)
    """
    (list of floats)
    total_time[rank] is the total wall time until that node finished its last
    call.  On slave i, only total_time[i] is available.
    """

    if not available:
        # dictionary for results:
        results = {}
        """
        (dictionary)
        if mpi is not available, the result of submit_call(..., id=a) will be
        cached in results[a] until get_result(a).
        """

    def submit_call(name_to_call, args=(), kwargs={},
                    module="__main__", time_est=1, id=None, slave=None):
        """
        Submit a call for parallel execution.

        If called by the master and slaves are available, the call is submitted
        to a slave for asynchronous execution.

        If called by a slave or if no slaves are available, the call is instead
        executed synchronously on this MPI node.

        **Examples:**

            1. Provide ids and time estimate explicitly:

               .. code-block:: python

                  for n in range(0,10):
                      mpi.submit_call("doit", (n,A[n]), id=n, time_est=n**2)

                  for n in range(0,10):
                      result[n] = mpi.get_result(n)

            2. Use generated ids stored in a list:

               .. code-block:: python

                  for n in range(0,10):
                      ids.append(mpi.submit_call("doit", (n,A[n])))

                  for n in range(0,10):
                      results.append(mpi.get_result(ids.pop()))

            3. Ignore ids altogether:

               .. code-block:: python

                  for n in range(0,10):
                      mpi.submit_call("doit", (n,A[n]))

                  for n in range(0,10):
                      results.append(mpi.get_next_result())

            4. Call a module function and use keyword arguments:

               .. code-block:: python

                  mpi.submit_call("solve", (), {"a":a, "b":b},
                      module="numpy.linalg")

            5. Call a static class method from a package:

               .. code-block:: python

                  mpi.submit_call("Network._get_histogram", (values, n_bins),
                      module="pyunicorn")

              Note that it is module="pyunicorn" and not
              module="pyunicorn.network" here.

        :arg str name_to_call: name of callable object (usually a function or
            static method of a class) as contained in the namespace specified
            by module.
        :arg tuple args: the positional arguments to provide to the callable
            object.  Tuples of length 1 must be written (arg,).  Default: ()
        :arg dict kwargs: the keyword arguments to provide to the callable
            object.  Default: {}
        :arg str module: optional name of the imported module or submodule in
            whose namespace the callable object is contained. For objects
            defined on the script level, this is "__main__", for objects
            defined in an imported package, this is the package name. Must be a
            key of the dictionary sys.modules (check there after import if in
            doubt).  Default: "__main__"
        :arg float time_est: estimated relative completion time for this call;
            used to find a suitable slave. Default: 1
        :type id: object or None
        :arg  id: unique id for this call. Must be a possible dictionary key.
            If None, a random id is assigned and returned. Can be re-used after
            get_result() for this is. Default: None
        :type slave: int > 0 and < mpi.size, or None
        :arg  slave: optional no. of slave to assign the call to. If None, the
            call is assigned to the slave with the smallest current total time
            estimate. Default: None
        :return object: id of call, to be used in get_result().
        """
        if id is None:
            id = numpy.random.uniform()
        if id in assigned:
            raise MPIException("id ", str(id), " already in queue!")
        if slave is not None and am_slave:
            raise MPIException(
                "only the master can use slave= in submit_call()")
        if slave is None or slave < 1 or slave >= size:
            # find slave with least estimated total time:
            slave = numpy.argmin(total_time_est)
        if available:
            # send name to call, args, time_est to slave:
            if _verbose:
                print(f"MPI master : assigning call with id {id} to slave "
                      f"{slave}: {name_to_call} {args} {kwargs} ...")
            comm.send((name_to_call, args, kwargs, module, time_est),
                      dest=slave)
        else:
            # do it myself right now:
            slave = 0
            if _verbose:
                print(f"MPI master : calling {name_to_call} {args} {kwargs} "
                      "...")
            try:
                object_to_call = eval(name_to_call,
                                      sys.modules[module].__dict__)
            except NameError:
                sys.stderr.write(str(sys.modules[module].__dict__.keys()))
                raise
            call_time = time.time()
            results[id] = object_to_call(*args, **kwargs)
            this_time = time.time() - call_time
            n_processed[0] += 1
            total_time[0] = time.time() - start_time
            stats.append({"id": id, "rank": 0,
                          "this_time": this_time,
                          "time_over_est": this_time / time_est,
                          "n_processed": n_processed[0],
                          "total_time": total_time[0]})

        total_time_est[slave] += time_est
        queue.append(id)
        slave_queue[slave].append(id)
        assigned[id] = slave
        return id

    def get_result(id):
        """
        Return result of earlier submitted call.

        Can only be called by the master.

        If the call is not yet finished, waits for it to finish.
        Results should be collected in the same order as calls were submitted.
        For each slave, the results of calls assigned to that slave must be
        collected in the same order as those calls were submitted.
        Can only be called once per call.

        :type id: object
        :arg  id: id of an earlier submitted call, as provided to or returned
                  by submit_call().

        :rtype:  object
        :return: return value of call.
        """
        source = assigned[id]
        if available:
            if slave_queue[source][0] != id:
                raise MPIException("get_result(" + str(id)
                                   + ") called before get_result("
                                   + str(slave_queue[source][0]) + ")!")
            if _verbose:
                print(f"MPI master : retrieving result for call with id {id} "
                      f"from slave {source} ...")
            (result, this_stats) = comm.recv(source=source)
            stats.append(this_stats)
            n_processed[source] = this_stats["n_processed"]
            total_time[source] = this_stats["total_time"]
        else:
            if _verbose:
                print(f"MPI master : returning result for call with id {id} "
                      "...")
            result = results[id]
            # TODO: rather return a copy and del the original?
        queue.remove(id)
        slave_queue[source].remove(id)
        assigned.pop(id)
        return result

    def get_next_result():
        """
        Return result of next earlier submitted call whose result has not yet
        been got.

        Can only be called by the master.

        If the call is not yet finished, waits for it to finish.

        :rtype:  object
        :return: return value of call, or None of there are no more calls in
                 the queue.
        """
        if len(queue) > 0:
            id = queue[0]
            return get_result(id)
        else:
            return None

    def info():
        """
        Print processing statistics.

        Can only be called by the master.
        """

        call_times = numpy.array([s["this_time"] for s in stats])
        call_quotients = numpy.array([s["time_over_est"] for s in stats])

        if available:
            slave_quotients = total_time/total_time_est
            print("\n"
                  "MPI: processing statistics\n"
                  "     =====================\n"
                  "     results collected:         "
                  f"{n_processed[1:].sum()}\n"
                  "     results not yet collected: "
                  f"{len(queue)}\n"
                  "     total reported time:       "
                  f"{call_times.sum()}\n"
                  "     mean time per call:        "
                  f"{call_times.mean()}\n"
                  "     std.dev. of time per call: "
                  f"{call_times.std()}\n"
                  "     coeff. of var. of actual over estd. time per call: "
                  f"{call_quotients.std()/call_quotients.mean()}\n"
                  "     slaves:                      "
                  f"{n_slaves}\n"
                  "     mean calls per slave:        "
                  f"{n_processed[1:].mean()}\n"
                  "     std.dev. of calls per slave: "
                  f"{n_processed[1:].std()}\n"
                  "     min calls per slave:         "
                  f"{n_processed[1:].min()}\n"
                  "     max calls per slave:         "
                  f"{n_processed[1:].max()}\n"
                  "     mean time per slave:        "
                  f"{total_time.mean()}\n"
                  "     std.dev. of time per slave: "
                  f"{total_time.std()}\n"
                  "     coeff. of var. of actual over estd. time per slave: "
                  f"{slave_quotients.std()/slave_quotients.mean()}\n")
        else:
            print("\n"
                  "MPI: processing statistics\n"
                  "     =====================\n"
                  "     results collected:         "
                  f"{n_processed[0]}\n"
                  "     results not yet collected: "
                  f"{len(queue)}\n"
                  "     total reported time:       "
                  f"{call_times.sum()}\n"
                  "     mean time per call:        "
                  f"{call_times.mean()}\n"
                  "     std.dev. of time per call: "
                  f"{call_times.std()}\n"
                  "     coeff. of var. of actual over estd. time per call: "
                  f"{call_quotients.std()/call_quotients.mean()}\n")

    def terminate():
        """
        Tell all slaves to terminate.

        Can only be called by the master.
        """
        global available
        if available:
            # tell slaves to terminate:
            for slave in range(1, size):
                if _verbose:
                    print(f"MPI master : telling slave {slave} "
                          "to terminate...")
                comm.send(("terminate", (), {}, "", 0), dest=slave)
            available = False

    def abort():
        """
        Abort execution on all MPI nodes immediately.

        Can be called by master and slaves.
        """
        traceback.print_exc()
        if _verbose:
            print("MPI master : aborting...")
        comm.Abort()

else:  # am_slave and available:

    total_time_est = numpy.zeros(size)*numpy.nan
    total_time_est[rank] = 0
    n_processed = numpy.zeros(size)*numpy.nan
    n_processed[rank] = 0
    total_time = numpy.zeros(size)*numpy.nan
    total_time[rank] = 0

    def serve():
        """
        Serve submitted calls until told to finish.

        Can only be called by slaves.

        Call this function from inside your definition of slave() if slaves
        need to perform initializations different from the master, like this:

        >>> def slave():
        >>>     do = whatever + initialization - is * necessary
        >>>     mpi.serve()
        >>>     do = whatever + cleanup - is * necessary

        If you don't define slave(), serve() will be called automatically by
        mpi.run().
        """
        if _verbose:
            print("MPI slave ", rank, ": waiting for calls.")
        # wait for orders:
        while True:
            # get next task from queue:
            (name_to_call, args, kwargs, module, time_est) = \
                comm.recv(source=0)
            # TODO: add some timeout and check whether master lives!
            if name_to_call == "terminate":
                if _verbose:
                    print("MPI slave", rank, ": terminating...")
                break
            if _verbose:
                print(f"MPI slave {rank}: calling {name_to_call} {args} ...")
            try:
                object_to_call = eval(name_to_call,
                                      sys.modules[module].__dict__)
            except NameError:
                sys.stderr.write(str(sys.modules[module].__dict__.keys()))
                raise
            total_time_est[rank] += time_est
            call_time = time.time()
            result = object_to_call(*args, **kwargs)
            this_time = time.time() - call_time
            n_processed[rank] += 1
            stats.append({"id": id, "rank": rank,
                          "this_time": this_time,
                          "time_over_est": this_time / time_est,
                          "n_processed": n_processed[rank],
                          "total_time": time.time() - start_time})
            if _verbose:
                print("MPI slave", rank, ": sending result...")
            comm.send((result, stats[-1]), dest=0)

    def abort():
        traceback.print_exc()
        if _verbose:
            print("MPI slave", rank, ": aborting...")
        comm.Abort()

    # TODO: update total_time_est at return time

_verbose = False


def run(verbose=False):
    """
    Run in master/slaves mode until master() finishes.

    Must be called on all MPI nodes after function master() was defined.

    On the master, run() calls master() and returns when master() returns.

    On each slave, run() calls slave() if that is defined, or calls serve()
    otherwise, and returns when slave() returns, or when master() returns on
    the master, or when master calls terminate().

    :arg bool verbose: whether processing information should be printed.
    """

    # transfer verbose into global environment:
    global _verbose
    _verbose = verbose
    """
    (bool) indicated whether processing information should be printed.
    """

    _globals = sys.modules['__main__'].__dict__

    if available:  # run in mpi mode
        if am_master:  # I'm master
            if verbose:
                print("MPI master : started, using", size-1, "slaves.")
            try:  # put everything in a try block to be able to terminate!
                if "master" in _globals:
                    _globals["master"]()
                else:
                    print("MPI master : function master() not found!")
            except ValueError:
                abort()
            terminate()
            if verbose:
                print("MPI master : finished.")
        else:  # I'm slave
            if "slave" in _globals:
                _globals["slave"]()
            else:
                serve()
    else:  # run as single processor
        if verbose:
            print("MPI master : not available, running as a single process.")
        if "master" in _globals:
            _globals["master"]()
        else:
            print("MPI master : function master() not found!")
        if verbose:
            print("MPI master : finished.")
