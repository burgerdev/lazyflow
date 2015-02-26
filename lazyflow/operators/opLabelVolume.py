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
#		   http://ilastik.org/license/
###############################################################################

from functools import partial
from abc import ABCMeta, abstractmethod, abstractproperty
import logging

import numpy as np
import vigra

from lazyflow.operator import Operator
from lazyflow.slot import InputSlot, OutputSlot
from lazyflow.rtype import SubRegion
from lazyflow.request import Request, RequestPool
from lazyflow.operators import OpCompressedCache, OpReorderAxes
from opLazyConnectedComponents import OpLazyConnectedComponents
from opImplementationChoice import OpImplementationChoice

logger = logging.getLogger(__name__)

# add your labeling implementations dynamically to this dict
# (vigra is added below)
labeling_implementations = {'lazy': OpLazyConnectedComponents}

## OpLabelVolume - the **unified** connected components operator
#
# This operator computes the connected component labeling of the input volume.
# The labeling is computed **seperately** per time slice and per channel.
class OpLabelVolume(Operator):

    name = "OpLabelVolume"

    ## provide the volume to label here
    # (arbitrary shape, dtype could be restricted, see the implementations'
    # property supportedDtypes below)
    Input = InputSlot()

    ## provide label that is treated as background
    Background = InputSlot(value=0)

    ## decide which CCL method to use
    #
    # currently available (see module variable labeling_implementations):
    # * 'vigra': use the fast algorithm from ukoethe/vigra
    # * 'lazy': use the lazy algorihm OpLazyConnectedComponents
    #
    # A change here deletes all previously cached results.
    Method = InputSlot(value='vigra')

    ## Labeled volume
    # Axistags and shape are the same as on the Input, dtype is an integer
    # datatype.
    # This slots behaviour depends on the labeling implementation used, see
    # their documentation for details. In general, it is advised to always use
    # CachedOutput to avoid recomputation, and this slot is only kept for
    # backwards compatibility.
    Output = OutputSlot()

    ## Cached label image
    # Axistags and shape are the same as on the Input, dtype is an integer
    # datatype.
    # This slot guarantees to give consistent results for subsequent requests.
    # For other behaviour, see the documentation of the underlying 
    # implementation.
    # This slot will be set dirty by time and channel if the background or the
    # input changes for the respective time-channel-slice.
    CachedOutput = OutputSlot()

    # cache access, see OpCompressedCache
    InputHdf5 = InputSlot(optional=True)
    CleanBlocks = OutputSlot()
    OutputHdf5 = OutputSlot()

    def __init__(self, *args, **kwargs):
        super(OpLabelVolume, self).__init__(*args, **kwargs)

        # we just want to have 5d data internally
        op5 = OpReorderAxes(parent=self)
        op5.Input.connect(self.Input)
        op5.AxisOrder.setValue('txyzc')
        self._op5 = op5

        self._opLabel = OpImplementationChoice(
            OpLabelingABC,
            parent=self,
            implementations=labeling_implementations,
            choiceSlot="Method")
        self._opLabel.Input.connect(self._op5.Output)
        self._opLabel.Background.connect(self.Background)
        self._opLabel.Method.connect(self.Method)
        self._opLabel.InputHdf5.connect(self.InputHdf5)

        self.CleanBlocks.connect(self._opLabel.CleanBlocks)
        self.OutputHdf5.connect(self._opLabel.OutputHdf5)

        self._op5_2 = OpReorderAxes(parent=self)
        self._op5_2_cached = OpReorderAxes(parent=self)
        self._op5_2.Input.connect(self._opLabel.Output)
        self._op5_2_cached.Input.connect(self._opLabel.CachedOutput)

        self.Output.connect(self._op5_2.Output)
        self.CachedOutput.connect(self._op5_2_cached.Output)

    def setupOutputs(self):

        # connect reordering operators
        self._op5_2.Input.connect(self._opLabel.Output)
        self._op5_2_cached.Input.connect(self._opLabel.CachedOutput)

        # set the final reordering operator's AxisOrder to that of the input
        origOrder = self.Input.meta.getAxisKeys()
        self._op5_2.AxisOrder.setValue(origOrder)
        self._op5_2_cached.AxisOrder.setValue(origOrder)

    def propagateDirty(self, slot, subindex, roi):
        if slot == self.Method:
            # We are changing the labeling method. In principle, the labelings
            # are equivalent, but not necessarily the same!
            self.Output.setDirty(slice(None))
        elif slot == self.Input:
            # handled by internal operator
            pass

    def setInSlot(self, slot, subindex, roi, value):
        assert slot == self.InputHdf5,\
            "Invalid slot for setInSlot(): {}".format( slot.name )
        # Nothing to do here.
        # Our Input slots are directly fed into the cache,
        #  so all calls to __setitem__ are forwarded automatically


## parent class for all connected component labeling implementations
class OpLabelingABC(Operator):
    __metaclass__ = ABCMeta

    ## input with axes 'txyzc'
    Input = InputSlot()

    ## background value
    Background = InputSlot(optional=True)

    Output = OutputSlot()
    CachedOutput = OutputSlot()

    # cache access, see OpCompressedCache
    InputHdf5 = InputSlot(optional=True)
    CleanBlocks = OutputSlot()
    OutputHdf5 = OutputSlot()

    # the numeric type that is used for labeling
    labelType = np.uint32

    ## list of supported dtypes
    @abstractproperty
    def supportedDtypes(self):
        pass

    def __init__(self, *args, **kwargs):
        super(OpLabelingABC, self).__init__(*args, **kwargs)
        self._cache = OpCompressedCache(parent=self)
        self._cache.name = "OpLabelVolume.OutputCache"
        self._cache.Input.connect(self.Output)
        self.CachedOutput.connect(self._cache.Output)
        self._cache.InputHdf5.connect(self.InputHdf5)
        self.OutputHdf5.connect(self._cache.OutputHdf5)
        self.CleanBlocks.connect(self._cache.CleanBlocks)

    def setupOutputs(self):

        # check if the input dtype is valid
        if self.Input.ready():
            dtype = self.Input.meta.dtype
            if dtype not in self.supportedDtypes:
                msg = "{}: dtype '{}' not supported "\
                    "with method 'vigra'. Supported types: {}"
                msg = msg.format(self.name, dtype, self.supportedDtypes)
                raise ValueError(msg)

        # set cache chunk shape to the whole spatial volume
        shape = np.asarray(self.Input.meta.shape, dtype=np.int)
        shape[0] = 1
        shape[4] = 1
        self._cache.BlockShape.setValue(tuple(shape))

        # setup meta for Output
        self.Output.meta.assignFrom(self.Input.meta)
        self.Output.meta.dtype = self.labelType

    def propagateDirty(self, slot, subindex, roi):
        # a change in either input or background makes the whole
        # time-channel-slice dirty (CCL is a global operation)
        outroi = roi.copy()
        outroi.start[1:4] = (0, 0, 0)
        outroi.stop[1:4] = self.Input.meta.shape[1:4]
        self.Output.setDirty(outroi)
        self.CachedOutput.setDirty(outroi)

    def setInSlot(self, slot, subindex, roi, value):
        assert slot == self.InputHdf5,\
            "Invalid slot for setInSlot(): {}".format( slot.name )
        # Nothing to do here.
        # Our Input slots are directly fed into the cache,
        #  so all calls to __setitem__ are forwarded automatically

    def execute(self, slot, subindex, roi, result):
        if slot == self.Output:
            # just label the ROI and write it to result
            self._label(roi, result)
        else:
            raise ValueError("Request to unknown slot {}".format(slot))

    def _label(self, roi, result):
        result = vigra.taggedView(result, axistags=self.Output.meta.axistags)
        bg = self.Background.value

        # do labeling in parallel over channels and time slices
        pool = RequestPool()

        start = np.asarray(roi.start, dtype=np.int)
        stop = np.asarray(roi.stop, dtype=np.int)
        for ti, t in enumerate(range(roi.start[0], roi.stop[0])):
            start[0], stop[0] = t, t+1
            for ci, c in enumerate(range(roi.start[4], roi.stop[4])):
                start[4], stop[4] = c, c+1
                newRoi = SubRegion(self.Output,
                                   start=tuple(start), stop=tuple(stop))
                resView = result[ti, ..., ci].withAxes(*'xyz')
                req = Request(partial(self._label3d, newRoi,
                                      bg, resView))
                pool.add(req)

        logger.debug(
            "{}: Computing connected components for ROI {} ...".format(
                self.name, roi))
        pool.wait()
        pool.clean()
        logger.debug("{}: Connected components computed.".format(
            self.name))

    ## compute the requested roi and put the results into result
    #
    # @param result the array to write into, 3d xyz
    @abstractmethod
    def _label3d(self, roi, bg, result):
        pass


## vigra connected components
class _OpLabelVigra(OpLabelingABC):
    name = "OpLabelVigra"
    supportedDtypes = [np.uint8, np.uint32, np.float32]

    def _label3d(self, roi, bg, result):
        source = vigra.taggedView(self.Input.get(roi).wait(),
                                  axistags='txyzc').withAxes(*'xyz')
        if source.shape[2] > 1:
            result[:] = vigra.analysis.labelVolumeWithBackground(
                source, background_value=int(bg))
        else:
            result[..., 0] = vigra.analysis.labelImageWithBackground(
                source[..., 0], background_value=int(bg))

labeling_implementations['vigra'] = _OpLabelVigra
