# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 15:23:49 2015

@author: burger
"""

import numpy as np
from matplotlib import pyplot as plot

s = np.asarray([4.05, 8.10, 12.35, 16.65])
r = np.asarray([1.11, 2.07, 3.04, 3.99])
p = np.asarray([1.92, 3.26, 4.70, 6.10])
m = np.asarray([1.48, ])

x = np.asarray([500, 1000, 1500, 2000])*1000*120*4/(1024.0**2)

plot.hold(True)

plot.plot(x,s)
plot.plot(x,r)
plot.plot(x,p)
a = x[0]
t = np.arange(1, 5)
plot.plot(t*a, t, 'k--')

plot.legend(("single thread", "requests", "multiprocess"), "upper left")
plot.xlabel("Data [MiB]")
plot.ylabel("Runtime [s]")

plot.show()
