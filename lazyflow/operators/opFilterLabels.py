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
from lazyflow.graph import Operator, InputSlot, OutputSlot

import numpy
import logging

logger = logging.getLogger(__name__)

filter_implementations = dict()


class OpFilterLabels(Operator):
    """
    Given a labeled volume, discard labels that have too few pixels.
    Zero is used as the background label
    """
    name = "OpFilterLabels"
    category = "generic"

    Input = InputSlot() 
    MinLabelSize = InputSlot(stype='int')
    MaxLabelSize = InputSlot(optional=True, stype='int')
    BinaryOut = InputSlot(optional=True, value = False, stype='bool')

    # choose between implementations
    # currently available: ['classic', 'lazy']
    Method = InputSlot(value='classic')
        
    Output = OutputSlot()

    def __init__(self, *args, **kwargs):
        super(OpFilterLabels, self).__init__(*args, **kwargs)
        self._configuredForMethod = None
        self._op = None

    def setupOutputs(self):
        method = self.Method.value
        if method not in filter_implementations.keys():
            raise ValueError("Unknown filter implementation "
                             "'{}'".format(method))

        cls = filter_implementations[method]
        if not issubclass(cls, OpFilterLabelsImpl):
            raise ValueError("Implementation must descend from "
                             "OpFilterLabelsImpl")

        if self._configuredForMethod == method:
            return
        self._configuredForMethod = method

        self._disconnectFilterOp()
        self._connectFilterOp(cls)

    def propagateDirty(self, slot, subindex, roi):
        # dirty handling is performed by implementation operators
        pass

    def _disconnectFilterOp(self):
        if self._op is None:
            return
        op = self._op
        self._op = None
        self.Output.disconnect()
        op.Input.disconnect()
        op.MinLabelSize.disconnect()
        op.MaxLabelSize.disconnect()
        op.BinaryOut.disconnect()

    def _connectFilterOp(self, cls):
        op = cls(parent=self)
        op.Input.connect(self.Input)
        op.MinLabelSize.connect(self.MinLabelSize)
        op.MaxLabelSize.connect(self.MaxLabelSize)
        op.BinaryOut.connect(self.BinaryOut)
        self.Output.connect(op.Output)
        self._op = op


class OpFilterLabelsImpl(Operator):

    Input = InputSlot() 
    MinLabelSize = InputSlot(stype='int')
    MaxLabelSize = InputSlot(optional=True, stype='int')
    BinaryOut = InputSlot(optional=True, value = False, stype='bool')
        
    Output = OutputSlot()
    
    def setupOutputs(self):
        self.Output.meta.assignFrom(self.Input.meta)
        
    def propagateDirty(self, inputSlot, subindex, roi):
        # all input slots can affect the entire output
        # FIXME why is this failing for BinaryOut?
        assert inputSlot == self.Input or inputSlot == self.MinLabelSize or inputSlot == self.MaxLabelSize
        self.Output.setDirty( slice(None) )
    


class OpFilterLabelsClassic(OpFilterLabelsImpl):

    ''' inherited slots
    Input = InputSlot() 
    MinLabelSize = InputSlot(stype='int')
    MaxLabelSize = InputSlot(optional=True, stype='int')
    BinaryOut = InputSlot(optional=True, value = False, stype='bool')
        
    Output = OutputSlot()
    '''
        
    def execute(self, slot, subindex, roi, result):
        minSize = self.MinLabelSize.value
        maxSize = None
        if self.MaxLabelSize.ready():
            maxSize = self.MaxLabelSize.value
        req = self.Input.get(roi)
        req.writeInto(result)
        req.wait()
        
        self.remove_wrongly_sized_connected_components(result, min_size=minSize, max_size=maxSize, in_place=True)
        return result

    def remove_wrongly_sized_connected_components(self, a, min_size, max_size, in_place):
        """
        Adapted from http://github.com/jni/ray/blob/develop/ray/morpho.py
        (MIT License)
        """
        bin_out = self.BinaryOut.value
        
        original_dtype = a.dtype
            
        if not in_place:
            a = a.copy()
        if min_size == 0 and (max_size is None or max_size > numpy.prod(a.shape)): # shortcut for efficiency
            if (bin_out):
                numpy.place(a,a,1)
            return a
        
        try:
            component_sizes = numpy.bincount( a.ravel() )
        except TypeError:
            # On 32-bit systems, must explicitly convert from uint32 to int
            # (This fix is just for VM testing.)
            component_sizes = numpy.bincount( numpy.asarray(a.ravel(), dtype=int) )
        bad_sizes = component_sizes < min_size
        if max_size is not None:
            numpy.logical_or( bad_sizes, component_sizes > max_size, out=bad_sizes )
        
        bad_locations = bad_sizes[a]
        a[bad_locations] = 0
        if (bin_out):
            # Replace non-zero values with 1
            numpy.place(a,a,1)
        return numpy.array(a, dtype=original_dtype)

filter_implementations['classic'] = OpFilterLabelsClassic
