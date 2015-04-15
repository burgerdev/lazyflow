# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 15:23:49 2015

@author: burger
"""

import numpy as np
from matplotlib import pyplot as plot

num_cores = 6

s = np.asarray([4.93, 9.75, 14.84, 19.80])
r = np.asarray([1.07, 2.14, 3.12, 4.41])
p = np.asarray([1.20, 2.24, 3.30, 4.26])
m = np.asarray([1.45, 2.28, 3.22, 4.81])

x = np.asarray([500, 1000, 1500, 2000])*1000*120*4/(1024.0**2)

plot.hold(True)

plot.plot(x, s/s[0]*100)
plot.plot(x, r/s[0]*100)
plot.plot(x, p/s[0]*100)
plot.plot(x, m/s[0]*100)
a = x[0]
t = np.arange(1, 5)
plot.plot(x, t*100/num_cores, 'k--')

plot.legend(("single thread", "requests", "multiprocess", "mpi", "theoretical optimum"), "upper left")
plot.xlabel("Data [MiB]")
plot.ylabel("Runtime [% of single threaded at 230MiB]")
plot.yticks(np.arange(0, 4.4, 1/3.0)*100)

plot.show()
