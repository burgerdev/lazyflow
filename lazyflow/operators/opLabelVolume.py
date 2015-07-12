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

import logging

import numpy as np

from lazyflow.operator import Operator
from lazyflow.slot import InputSlot, OutputSlot
from lazyflow.operators import OpCompressedCache
from opSimpleConnectedComponents import OpSimpleConnectedComponents
from opLazyConnectedComponents import OpLazyConnectedComponents
from opImplementationChoice import OpImplementationChoice

from lazyflow.roi import determineBlockShape

logger = logging.getLogger(__name__)

# add your labeling implementations dynamically to this dict
# (see below the operators for examples)
labeling_implementations = {}


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
    # * 'lazy': use the lazy algorithm OpLazyConnectedComponents
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

    def __new__(cls, *args, **kwargs):
        kwargs['implementations'] = labeling_implementations
        kwargs['choiceSlot'] = 'Method'
        op = OpImplementationChoice(OpCachedLabeling,
                                    *args, **kwargs)
        op.Method.setValue('vigra')
        return op


class OpCachedLabeling(Operator):
    """
    abstract class for labeling with cache
    """

    name = "OpCachedLabeling"

    ## provide the volume to label here
    # (arbitrary shape, dtype could be restricted, see the implementations'
    # property supportedDtypes below)
    Input = InputSlot()

    ## provide label that is treated as background
    Background = InputSlot(value=0)

    ## Labeled volume
    # Axistags and shape are the same as on the Input, dtype is an integer
    # datatype.
    # This slot computes the connected component image only for the
    # requested roi, no consistency guarantees apply! We keep this slot
    # for backwards compatibility.
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
        super(OpCachedLabeling, self).__init__(*args, **kwargs)
        self.__labeler = self._getLabeler()
        self.__labeler.Input.connect(self.Input)
        self.__labeler.Background.connect(self.Background)
        self.Output.connect(self.__labeler.Output)
        self.__cache = None

    def setupOutputs(self):
        self.__resetCache()

    def __resetCache(self):
        if self.__cache is not None:
            # disconnect and remove old cache
            self.CachedOutput.disconnect()
            self.OutputHdf5.disconnect()
            self.CleanBlocks.disconnect()
            self.__cache.InputHdf5.disconnect()
            self.__cache.Input.disconnect()
            self.__cache = None

        block_shape = self._getBlockShape()

        # set up new cache
        self.__cache = OpCompressedCache(parent=self)
        self.__cache.BlockShape.setValue(block_shape)
        self.CachedOutput.connect(self.__cache.Output)
        self.OutputHdf5.connect(self.__cache.OutputHdf5)
        self.CleanBlocks.connect(self.__cache.CleanBlocks)
        self.__cache.InputHdf5.connect(self.InputHdf5)
        self.__cache.Input.connect(self.__labeler.Output)

    def _getBlockShape(self):
        # set time, channel shape to 1
        ts = self.Input.meta.getTaggedShape()
        spatial = []
        spatial_keys = []
        for k in ts.keys():
            if k in 'tc':
                ts[k] = 1
            else:
                spatial.append(ts[k])
                spatial_keys.append(k)
        max_shape = tuple(ts[k] for k in ts)

        if self.Input.meta.ideal_blockshape is not None:
            shape = self.Input.meta.ideal_blockshape
            shape = np.where(shape > 0, shape, max_shape)
            shape = np.max(shape, max_shape)
        else:
            # go for blocks of roughly 10MiB size (e.g. 1024x1024x10)
            shape = determineBlockShape(max_shape, 10*1024**2)
        return shape

    def execute(self, slot, subindex, roi, result):
        raise RuntimeError("Internal connections are wrong")

    def propagateDirty(self, slot, subindex, roi):
        # dirty propagation is handled by wrapped operator
        pass

    def setInSlot(self, slot, subindex, roi, value):
        msg = "Invalid slot for setInSlot(): {}".format(slot.name)
        if slot is not self.InputHdf5:
            raise ValueError(msg)
        elif self.__cache is None:
            msg = "Can't set data to slot: it is not configured yet."
            raise RuntimeError(msg)
        # Nothing to do here.
        # Our Input slots are directly fed into the cache,
        #  so all calls to __setitem__ are forwarded automatically


class OpLazyCC(OpCachedLabeling):
    """
    lazy labeling class
    """

    name = "OpLazyCC"

    def _getLabeler(self):
        return OpLazyConnectedComponents(parent=self)

    def _getBlockShape(self):
        # set time, channel shape to 1
        ts = self.Input.meta.getTaggedShape()
        spatial = []
        spatial_keys = []
        for k in ts.keys():
            if k in 'tc':
                ts[k] = 1
            else:
                spatial.append(ts[k])
                spatial_keys.append(k)
        max_shape = tuple(ts[k] for k in ts)

        if self.Input.meta.ideal_blockshape is not None:
            shape = self.Input.meta.ideal_blockshape
            shape = np.where(shape > 0, shape, max_shape)
            print(shape, max_shape)
            shape = np.maximum(shape, max_shape)
        else:
            # go for blocks of roughly 10MiB size (e.g. 1024x1024x10)
            shape = determineBlockShape(max_shape, 10*1024**2)
        return shape

# add lazy labeling to available implementations
labeling_implementations['lazy'] = OpLazyCC


class OpNonLazyCC(OpCachedLabeling):
    """
    Non-lazy labeling class, which uses OpSimpleConnectedComponents.
    Consistency is achieved by setting the OpCompressedCache.BlockShape to the
    entire spatial shape, leaving t and c at 1.
    """

    name = "OpNonLazyCC"

    def _getLabeler(self):
        return OpSimpleConnectedComponents(parent=self)

    def _getBlockShape(self):
        # set time, channel shape to 1
        ts = self.Input.meta.getTaggedShape()
        for k in ts.keys():
            if k in 'tc':
                ts[k] = 1
        block_shape = tuple(ts[k] for k in ts)
        return block_shape

# add non-lazy labeling to available implementations
labeling_implementations['vigra'] = OpNonLazyCC
