import numpy as np
from pyunicorn.timeseries import RecurrenceNetwork

x = np.sin(np.linspace(0, 10 * np.pi, 1000))
net = RecurrenceNetwork(x, recurrence_rate=0.05)
print(net.transitivity())
