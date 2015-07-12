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

import threading
import itertools
import multiprocessing
import time

import numpy as np
from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.roi import roiToSlice


# Some tools to aid automated testing

# check that two label images have the same meaning
# (label images can differ in the actual labels, but cannot differ in their
# spatial structure)
#
# Examples:
#   Input:
#       1 1 0    250 250 0
#       1 1 0    250 250 0
#       0 0 0    0   0   0
#   Output: None
#
#   Input:
#       1 1 0    250 250 0
#       1 1 0    1   1   0
#       0 0 0    0   0   0
#   Output: AssertionError
#
def assertEquivalentLabeling(labelImage, referenceImage):
    x = labelImage
    y = referenceImage
    assert np.all(x.shape == y.shape),\
        "Shapes do not agree ({} vs {})".format(x.shape, y.shape)

    # identify labels used in x
    labels = set(x.flat)
    for label in labels:
        if label == 0:
            continue
        idx = np.where(x == label)
        refblock = y[idx]
        # check that labels are the same
        corner = [a[0] for a in idx]
        print("Inspecting region of size {} at {}".format(refblock.size, corner))

        assert np.all(refblock == refblock[0]),\
            "Uniformly labeled region at coordinates {} has more than one label in the reference image".format(corner)
        # check that nothing else is labeled with this label
        m = refblock.size
        n = (y == refblock[0]).sum()
        assert m == n, "There are more pixels with (reference-)label {} than pixels with label {}.".format(refblock[0], label)

    assert len(labels) == len(set(y.flat)), "The number of labels does not agree, perhaps some region was missed"


class OpArrayPiperWithAccessCount(Operator):
    """
    array piper that counts how many times its execute function has been called
    """
    Input = InputSlot(allow_mask=True)
    Output = OutputSlot(allow_mask=True)

    def __init__(self, *args, **kwargs):
        super(OpArrayPiperWithAccessCount, self).__init__(*args, **kwargs)
        self.clear()
        self._lock = threading.Lock()

    def setupOutputs(self):
        self.Output.meta.assignFrom(self.Input.meta)

    def execute(self, slot, subindex, roi, result):
        with self._lock:
            self.accessCount += 1
            self.requests.append(roi)
        req = self.Input.get(roi)
        req.writeInto(result)
        req.block()

    def propagateDirty(self, slot, subindex, roi):
        self.Output.setDirty(roi)

    def clear(self):
        self.requests = []
        self.accessCount = 0


class OpBigArraySimulator(Operator):
    """
    operator that simulates a big region, useful for testing lazy ops

    The operator takes an input arry, which is then repeated so that it
    matches Shape. Note that there is no dirty propagation.
    """
    Input = InputSlot(allow_mask=True)
    Shape = InputSlot()
    Output = OutputSlot(allow_mask=True)

    def __init__(self, *args, **kwargs):
        super(OpBigArraySimulator, self).__init__(*args, **kwargs)

    def setupOutputs(self):
        self.Output.meta.assignFrom(self.Input.meta)
        shape = self.Shape.value
        assert len(shape) == len(self.Input.meta.shape)
        self.Output.meta.shape = shape
        self.Output.meta.ideal_blockshape = self.Input.meta.shape

    def execute(self, slot, subindex, roi, result):
        vol = self.Input[...].wait()
        blockshape = np.asarray(vol.shape, dtype=np.int)
        start = np.asarray(roi.start, dtype=np.int)
        stop = np.asarray(roi.stop, dtype=np.int)
        shape = stop - start

        # roll the array such that it is aligned with roi.start
        for d in range(len(start)):
            vol = np.roll(vol, -(start[d] % blockshape[d]), axis=d)

        # determine numberof blocks in each direction
        not_fitting = np.mod(shape, blockshape) > 0
        numblocks = shape / blockshape
        numblocks += np.where(not_fitting, 1, 0)

        ranges = [xrange(numblocks[d]) for d in range(len(numblocks))]
        ind_it = itertools.imap(np.asarray, itertools.product(*ranges))
        for inds in ind_it:
            local_start = inds * blockshape
            local_stop = np.minimum(local_start + blockshape, shape)
            local_shape = local_stop - local_start
            res_slice = roiToSlice(local_start, local_stop)
            vol_slice = roiToSlice(np.zeros_like(local_shape),
                                   local_shape)
            result[res_slice] = vol[vol_slice]

    def propagateDirty(self, slot, subindex, roi):
        pass


class Timeout(object):
    class TimeoutError(Exception):
        pass

    def __init__(self, seconds, function):
        self.__seconds = seconds
        self.__condition = threading.Condition()
        def wrapped():
            function()
            with self.__condition:
                self.__condition.notifyAll()
        self.__function = wrapped

    def __on_timeout(self):
        msg = "waited {:.1f}s".format(self.__seconds)
        raise self.TimeoutError(msg)

    def start(self):
        executor = threading.Thread(target=self.__function)
        executor.daemon = True
        executor.start()
        with self.__condition:
            self.__condition.wait(self.__seconds)
        if executor.isAlive():
            self.__on_timeout()
        else:
            executor.join()


class OpCallWhenDirty(Operator):
    """
    calls the attribute 'function' when Input gets dirty

    The parameters of the dirty call are stored in attributres.
    """

    Input = InputSlot(allow_mask=True)
    Output = OutputSlot(allow_mask=True)

    function = lambda: None
    slot = None
    roi = None

    def setupOutputs(self):
        self.Output.meta.assignFrom(self.Input.meta)

    def execute(self, slot, subindex, roi, result):
        req = self.Input.get(roi)
        req.writeInto(result)
        req.block()

    def propagateDirty(self, slot, subindex, roi):
        try:
            self.slot = slot
            self.subindex = subindex
            self.roi = roi
            self.function()
        except:
            raise
        finally:
            self.Output.setDirty(roi)
