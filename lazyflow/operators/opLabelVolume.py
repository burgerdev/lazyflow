
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

        # we just want to have 5d data internally
        op5 = OpReorderAxes(parent=self)
        op5.Input.connect(self.Input)
        op5.AxisOrder.setValue('xyzct')
        self._op5 = op5

        self._opLabel = OpLabel5D(parent=self)
        self._opLabel.Input.connect(op5.Output)
        self._opLabel.Method.connect(self.Method)

        op5_2 = OpReorderAxes(parent=self)
        op5_2.Input.connect(self._opLabel.Output)
        #op5_2.AxisOrder.setValue("".join([ax for ax in self.Input.meta.getTaggedShape()]))
        self._op5_2 = op5_2

        self.Output.connect(op5_2.Output)
        #FIXME connect to the right slot
        self.CachedOutput.connect(op5_2.Output)

    def setupOutputs(self):
        self._op5_2.AxisOrder.setValue(
            "".join([s for s in self.Input.meta.getTaggedShape()]))
        pass

    def propagateDirty(self, slot, subindex, roi):
        # nothing to do here, all slots are connected
        pass


class OpLabel5D(Operator):
    Input = InputSlot()
    Method = InputSlot()
    Output = OutputSlot()
    CachedOutput = OutputSlot()
    def __init__(self, *args, **kwargs):
        super(OpLabel5D, self).__init__(*args, **kwargs)
        self._cache = None
        self._metaProvider = _OpMetaProvider(parent=self)

    def setupOutputs(self):
        labelType = np.int32

        # remove unneeded old cache
        if self._cache is not None:
            self._cache.Input.disconnect()
            self._cache = None

        m = self.Input.meta
        self._metaProvider.setMeta(
            MetaDict({'shape': m.shape, 'dtype': labelType,
                      'axistags': m.axistags}))

        self._cache = OpCompressedCache(parent=self)
        self._cache.Input.connect(self._metaProvider.Output)
        self.Output.meta.assignFrom(self._cache.Output.meta)
        self.CachedOutput.meta.assignFrom(self._cache.Output.meta)

        s = self.Input.meta.getTaggedShape()
        self._cached = np.zeros((s['c'], s['t']))

    def propagateDirty(self, slot, subindex, roi):
        # a change somewhere makes the whole time-channel-slice dirty
        # (CCL with vigra is a global operation)
        for t in range(roi.start[4], roi.stop[4]):
            for c in range(roi.start[3], roi.stop[3]):
                self._cached[c, t] = 0

    def execute(self, slot, subindex, roi, result):
        
        # do labeling in parallel over channels and time slices
        #FIXME make parallel
        for t in range(roi.start[4], roi.stop[4]):
            for c in range(roi.start[3], roi.stop[3]):
                # update the whole slice, if needed
                print("Updating slice c={}, t={}".format(c,t))
                self._updateSlice(c, t)
        
        req = self._cache.Output.get(roi)
        req.writeInto(result)
        req.block()
    
    def _updateSlice(self, c, t):
        if self._cached[c, t]:
            return
        method = self.Method[0].wait()
    
        if method == 'vigra':
            cc = self._vigraCC
        elif method == 'blocked':
            cc = self._blockedCC
        else:
            raise ValueError("Connected Component Labeling method '{}' not supported".format(method))
        
        cc(c, t)
        #FIXME add Lock
        self._cached [c,t] = 1

    def _vigraCC(self, c, t):
        source = vigra.taggedView(self.Input[..., c, t].wait(), axistags='xyzct')
        source = source.withAxes(*'xyz')
        #FIXME hardcoded background
        result = vigra.analysis.labelVolumeWithBackground(
            source, background_value=0)
        result = result.withAxes(*'xyzct')

        stop = np.asarray(self.Input.meta.shape)
        start = 0*stop
        start[3:] = (c, t)
        stop[3:] = (c+1, t+1)
        roi = SubRegion(self._cache.Input, start=start, stop=stop)

        self._cache.setInSlot(self._cache.Input, (), roi, result)

    def _blockedCC(self, c, t):
        raise NotImplementedError()



'''
## Prototype of blockedarray integration
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
        print(roi)
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


