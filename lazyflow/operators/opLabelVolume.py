
from threading import Lock as ThreadLock

import numpy as np
import vigra

from lazyflow.operator import Operator
from lazyflow.operatorWrapper import OperatorWrapper
from lazyflow.slot import InputSlot, OutputSlot
from lazyflow.rtype import SubRegion
from lazyflow.metaDict import MetaDict
from lazyflow.request import Request, RequestPool
from lazyflow.operators import OpArrayCache, OpCompressedCache, OpReorderAxes
from lazyflow.operators import OpMultiArraySlicer, OpMultiArrayStacker

# try to import the blockedarray module, fail only if neccessary
try:
    import blockedarray
except ImportError as e:
    _blockedarray_module_available = False
    _importMsg = str(e)
else:
    _blockedarray_module_available = True

## OpLabelVolume - the **unified** connected components operator
#
# This operator computes the connected component labeling of the input volume.
# The labeling is computed **seperately** per time slice and per channel.
class OpLabelVolume(Operator):

    ## provide the volume to label here
    # (arbitrary shape, dtype could be restricted #TODO)
    Input = InputSlot()
 
    ## provide labels that are treated as background
    # the shape of the background albels must match the shape of the volume in
    # channel and in time axis, and must have singleton spatial axes.
    # E.g.: volume.taggedShape = {'x': 10, 'y': 12, 'z': 5, 'c': 3, 't': 100}
    # ==>
    # background.taggedShape = {'c': 3, 't': 100}
    #TODO not implemented yet!
    #Background = InputSlot(optional=True)

    ## decide which CCL method to use
    #
    # currently available:
    # * 'vigra': use the fast algorithm from ukoethe/vigra
    # * 'blocked': use the memory saving algorithm from thorbenk/blockedarray
    Method = InputSlot(value=np.asarray(('vigra',), dtype=np.object))

    ## Labeled volume
    # Axistags and shape are the same as on the Input, dtype is an integer
    # datatype.
    # Note: The output might be cached internally depending on the chosen CC
    # method, use the CachedOutput to avoid duplicate caches. (see inner 
    # operator below for details)
    Output = OutputSlot()

    ## Cached label image
    # Access the internal cache. If you were planning to cache the labeled
    # volume, be sure to use this slot, since it makes use of the internal
    # cache that might be in use anyway.
    CachedOutput = OutputSlot()

    def __init__(self, *args, **kwargs):
        super(OpLabelVolume, self).__init__(*args, **kwargs)
        # connect the operators as far as we can without knowledge of the input

        # we just want to have 5d data internally
        op5 = OpReorderAxes(parent=self)
        op5.Input.connect(self.Input)
        op5.AxisOrder.setValue('xyzct')
        self._op5 = op5

        # slice input along channel and time axis
        self._timeSlicer = OpMultiArraySlicer(parent=self)
        self._timeSlicer.AxisFlag.setValue('t')
        self._timeSlicer.Input.connect(self._op5.Output)
        assert self._timeSlicer.Slices.level == 1

        self._channelSlicer = OperatorWrapper(OpMultiArraySlicer, parent=self)
        self._channelSlicer.AxisFlag.setValue( 'c' )
        self._channelSlicer.Input.connect(self._timeSlicer.Slices )
        assert self._channelSlicer.Slices.level == 2

        # doubly wrap inner operator
        opCC = OperatorWrapper(_OpWrappedSwitchLabelingImplementations,
                               parent=self, broadcastingSlotNames=['Method'])
        assert opCC.Input.level == 2
        assert opCC.Output.level == 2

        #TODO slice background values

        opCC.Input.connect(self._channelSlicer.Slices)
        self._opCC = opCC

        # stack output along channel and time axis again
        self._channelStacker = OperatorWrapper(OpMultiArrayStacker, parent=self)
        self._channelStacker.AxisFlag.setValue('c')
        self._channelStacker.AxisIndex.setValue(3)
        assert self._channelStacker.Images.level == 2
        self._channelStacker.Images.connect(self._opCC.Output)

        self._timeStacker = OpMultiArrayStacker(parent=self)
        self._timeStacker.AxisFlag.setValue('t')
        self._timeStacker.AxisIndex.setValue(4)
        assert self._channelStacker.Output.level == 1
        assert self._timeStacker.Images.level == 1
        self._timeStacker.Images.connect(self._channelStacker.Output)

        op5_2 = OpReorderAxes(parent=self)
        op5_2.Input.connect(self._timeStacker.Output)
        self._op5_2 = op5_2

        self.Output.connect(op5_2.Output)

    def setupOutputs(self):
        # now we know what the input looks like, and which method to use, so we
        # can set up the caching and the background labels

        if False and self.Background.ready():  #TODO remove when implemented
            self._opCC.Background.connect(self._opBgSlice)
        else:
            val = np.asarray([0])
            val = vigra.taggedView(val, axistags='x')
            # set all background values to 0
            for tSlot in self._opCC.Background:
                for cSlot in tSlot:
                    cSlot.setValue(val)

        # decide whether an outer cache is needed
        if self._opCC[0][0].usesInternalCache():
            self.CachedOutput.connect(self._op5_2.Output)
        else:
            #FIXME ArrayCache or CompressedCache?
            cache = OpCompressedCache(parent=self)
            cache.Input.connect(self._op5_2.Output)
            self._cache = cache
            self.CachedOutput.connect(cache.Output)

        self._op5_2.AxisOrder.setValue(
            "".join([s for s in self.Input.meta.getTaggedShape()]))

    def propagateDirty(self, slot, subindex, roi):
        pass


## Wrapper for multiple CL implementations
#
# This wrapper decides which implementation of connected component labeling
# should be used and how caching is handled. There are two main ways of doing
# CCL:
#   * LAZY: compute the labeling on only that part that is really relevant for
#           providing a unique labeling for other regions in the future
#   * FULL: when the first request comes in, compute the labeling once and
#           write it to a cache
class _OpSwitchLabelingImplementations(Operator):

    ## the volume to label
    # (3d with axes 'xyz', dtype ??? #TODO)
    Input = InputSlot()

    ## provide labels that are treated as background
    # a single element array
    Background = InputSlot()

    ## decide which CCL method to use
    #
    # This slot is the whole point of this operator
    Method = InputSlot(value=np.asarray(('vigra',), dtype=np.object))

    Output = OutputSlot()

    _ccl = None

    def setupOutputs(self):

        method = self.Method[0].wait()

        # disconnect internal operator if we created one earlier
        if self._ccl is not None:
            self.Output.disconnect()
            self._ccl.Input.disconnect()
            self._ccl.Background.disconnect()
            self._ccl = None

        # switch the implementation
        if method == 'vigra':
            self._ccl = _OpVigraCCL(parent=self)
        elif method == 'blocked':
            self._ccl = _OpBlockedArrayCCL(parent=self)
        else:
            raise ValueError(
                "Unknown connected components labeling method '{}'".format(
                    method))
        self._ccl.Input.connect(self.Input)
        self._ccl.Background.connect(self.Background)
        self.Output.connect(self._ccl.Output)

    def execute(self, slot, subindex, roi, result):
        assert False, "Shouldn't get here"

    def propagateDirty(self, slot, subindex, roi):
        pass

    def usesInternalCache(self):
        if self._ccl is not None:
            return self._ccl.usesInternalCache()
        else:
            return True

## wrapped operator
# currently double wrapping not available
class _OpWrappedSwitchLabelingImplementations(Operator):

    ## the volume to label
    # (3d with axes 'xyz', dtype ??? #TODO)
    Input = InputSlot(level=1)

    ## provide labels that are treated as background
    # a single element array
    Background = InputSlot(level=1)

    ## decide which CCL method to use
    #
    # This slot is the whole point of this operator
    Method = InputSlot(value=np.asarray(('vigra',), dtype=np.object))

    Output = OutputSlot(level=1)

    def __init__(self, *args, **kwargs):
        super(_OpWrappedSwitchLabelingImplementations, self).__init__(*args, **kwargs)
        op = OperatorWrapper(_OpSwitchLabelingImplementations, parent=self,
                             broadcastingSlotNames=['Method'])
        op.Input.connect(self.Input)
        op.Background.connect(self.Background)
        op.Method.connect(self.Method)
        self.Output.connect(op.Output)
        self._op = op

    def execute(self, slot, subindex, roi, result):
        pass

    def propagateDirty(self, slot, subindex, roi):
        pass

    def __getitem__(self, key):
        return self._op[key]


class _OpCCLInterface(Operator):
    Input = InputSlot()
    Background = InputSlot()
    Output = OutputSlot()

    def usesInternalCache(self):
        raise NotImplementedError("Abstract method not implemented")


class _OpVigraCCL(_OpCCLInterface):

    def usesInternalCache(self):
        return False

    def setupOutputs(self):
        assert self.Input.meta.dtype in [np.uint8, np.uint32],\
            "Datatype {} not supported".format(self.Input.meta.dtype)
        self.Output.meta.assignFrom(self.Input.meta)
        self.Output.meta.dtype = np.int32

    def execute(self, slot, subindex, roi, result):
        s = self.Input.meta.shape
        source = self.Input[...].wait()
        bg = self.Background[0].wait()
        temp = vigra.analysis.labelVolumeWithBackground(
            source, background_value=int(bg[0]))
        result[:] = temp[roi.toSlice()]

    def propagateDirty(self, dirtySlot, subindex, roi):
        # a change to any of our two slots requires recomputation
        self.Output.setDirty(slice(None))


class _OpFullCCLInterface(_OpCCLInterface):

    def usesInternalCache(self):
        return True
    
    def __init__(self, *args, **kwargs):
        super(_OpFullCCLInterface, self).__init__(*args, **kwargs)
        self._lock = ThreadLock()
        self._alreadyCached = False
        self._cache = None
        self._metaProvider = _OpMetaProvider(parent=self)
    
    def setupOutputs(self):
        assert len(self.Input.meta.shape) == 3, "Only 3d data supported"

        # remove unneeded old cache
        if self._cache is not None:
            self._cache.Input.disconnect()
            self._cache = None

        m = self.Input.meta
        self._metaProvider.setMeta(
            MetaDict({'shape': m.shape, 'dtype': np.int32,
                      'axistags': m.axistags}))

        self._cache = OpCompressedCache(parent=self)
        self._cache.Input.connect(self._metaProvider.Output)
        self.Output.meta.assignFrom(self._cache.Output.meta)

    def execute(self, slot, subindex, roi, result):
        # the first call to the output slot triggers computation, all others
        # wait for the data to arrive
        self._lock.acquire()
        if not self._alreadyCached:
            self._computeCCL()
            self._alreadyCached = True
        self._lock.release()

        # return the requested roi
        req = self._cache.Output.get(roi)
        req.writeInto(result)
        req.block()

    def propagateDirty(self, dirtySlot, subindex, roi):
        # a change to any of our two slots requires recomputation
        self._alreadyCached = False
        self._cache.Input.setDirty(slice(None))

    def _computeCCL(self):
        raise NotImplementedError("This method is still abstract")


class _OpBlockedArrayCCL(_OpFullCCLInterface):
    def setupOutputs(self):
        super(_OpBlockedArrayCCL, self).setupOutputs()
        assert self.Input.meta.dtype in [np.uint8],\
            "Datatype {} not supported".format(self.Input.meta.dtype)
        assert _blockedarray_module_available,\
            "Failed to import blockedarray: {}".format(_importMsg)

    def _computeCCL(self):
        source = _Source(self._op5.Output, blockShape, c, t)
        sink = _Sink(self._cache, c, t)
        cc = blockedarray.dim3.ConnectedComponents(source, blockShape)
        cc.writeToSink(sink)


## An OperatorWrapper to promote slots to arbitrary levels
#
# Level==1 -> ordinary operator wrapper
# FIXME just a simple implementation, many details missing
'''
class _OperatorMultiWrapper(OperatorWrapper):
    name = "OperatorMultiWrapper"
    def __init__(self, level, operatorClass, operator_args=None,
                 operator_kwargs=None, parent=None, graph=None,
                 promotedSlotNames=None, broadcastingSlotNames=None):
        if level > 1:
            print("Constructing outer wrapper")
            # wrap this class recursively
            super(_OperatorMultiWrapper, self).__init__(
                _OperatorMultiWrapper,
                operator_args=[level-1, operatorClass],
                operator_kwargs={},
                parent=parent, graph=graph)
        else:
            print("constructing inner wrapper")
            # construct a simple OperatorWrapper
            super(_OperatorMultiWrapper, self).__init__(
                operatorClass, operator_args=operator_args,
                operator_kwargs=operator_kwargs, parent=parent,
                graph=graph, promotedSlotNames=promotedSlotNames,
                broadcastingSlotNames=broadcastingSlotNames)
'''

## Feeds meta data into OpCompressedCache
#
# This operator is needed because we
#   - don't connect OpCompressedCache directly to a real InputSlot
#   - feed data to cache by setInSlot()
class _OpMetaProvider(Operator):
    Output = OutputSlot()

    def __init__(self, *args, **kwargs):
        # Configure output with given metadata.
        super(_OpMetaProvider, self).__init__(*args, **kwargs)

    def setupOutputs(self):
        pass

    def execute(self, slot, subindex, roi, result):
        assert False,\
            "The cache asked for data which should not happen."

    def setMeta(self, meta):
        self.Output.meta.assignFrom(meta)


if _blockedarray_module_available:

    class _Source(blockedarray.adapters.SourceABC):
        def __init__(self, slot, blockShape):
            super(_Source, self).__init__()
            self._slot = slot
            self._blockShape = blockShape
            self._p = np.asarray(slot.meta.shape, dtype=np.long)*0
            self._q = np.asarray(slot.meta.shape, dtype=np.long)

        def pySetRoi(self, roi):
            assert len(roi) == 2
            self._p = np.asarray(roi[0], dtype=np.long)
            self._q = np.asarray(roi[1], dtype=np.long)

        def pyShape(self):
            return tuple(self._slot.meta.shape)

        def pyReadBlock(self, roi, output):
            assert len(roi) == 2
            roiP = np.asarray(roi[0])
            roiQ = np.asarray(roi[1])
            p = self._p + roiP
            q = p + roiQ - roiP
            if np.any(q > self._q):
                raise IndexError("Requested roi is too large for selected "
                                "roi (previous call to setRoi)")
            sub = SubRegion(self._slot, start=p, stop=q)
            req = self._slot.get(sub)
            req.writeInto(output)
            req.block()
            return True

    class _Sink(blockedarray.adapters.SinkABC):
        def __init__(self, op):
            super(_Sink, self).__init__()
            self._op = op

        def pyWriteBlock(self, roi, block):
            assert len(roi) == 2
            sub = SubRegion(self._op.Input, start=roi[0], stop=roi[1])
            self._op.setInSlot(self._op.Input, (), sub, block)


