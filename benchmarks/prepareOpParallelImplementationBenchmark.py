###############################################################################
#   lazyflow: data flow based lazy parallel computation framework
#
#       Copyright (C) 2011-2014, the ilastik developers
#                                <team@ilastik.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the Lesser GNU General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# See the files LICENSE.lgpl2 and LICENSE.lgpl3 for full text of the
# GNU Lesser General Public License version 2.1 and 3 respectively.
# This information is also available on the ilastik web site at:
#          http://ilastik.org/license/
###############################################################################

import vigra
import numpy as np

if __name__ == "__main__":
    for dim_x in (500, 1000, 1500, 2000):
        shape = 1, dim_x, 1000, 120, 1
        chunkShape = 1, 500, 500, 20, 1
        x = np.random.randint(0, 255, size=shape).astype(np.float32)
        x = vigra.taggedView(x, axistags='txyzc')

        vigra.writeHDF5(x, '/tmp/test_{}.h5'.format(dim_x), '/data')
