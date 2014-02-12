
from threading import Lock as ThreadLock
from functools import partial

import numpy as np
import vigra

from lazyflow.operator import Operator
from lazyflow.slot import InputSlot, OutputSlot
from lazyflow.rtype import SubRegion
from lazyflow.metaDict import MetaDict
from lazyflow.request import Request, RequestPool
from lazyflow.operators import OpCompressedCache, OpReorderAxes

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
    #TODO relax requirements (single value is already working)
    Background = InputSlot(optional=True)

    ## decide which CCL method to use
    #
    # currently available:
    # * 'vigra': use the fast algorithm from ukoethe/vigra
    # * 'blocked': use the memory saving algorithm from thorbenk/blockedarray
    Method = InputSlot(value='vigra')

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

        self._opLabel = None

        self._op5_2 = OpReorderAxes(parent=self)
        self._op5_2_cached = OpReorderAxes(parent=self)

        self.Output.connect(self._op5_2.Output)
        self.CachedOutput.connect(self._op5_2_cached.Output)
        
        # available OpLabelingABCs:
        self._labelOps = {'vigra': _OpLabelVigra, 'blocked': _OpLabelBlocked}

    def setupOutputs(self):

        if self._opLabel is not None:
            # fully remove old labeling operator
            self._op5_2.Input.disconnect()
            self._op5_2_cached.Input.disconnect()
            self._opLabel.Input.disconnect()
            del self._opLabel

        self._opLabel = self._labelOps[self.Method.value](parent=self)
        self._opLabel.Input.connect(self._op5.Output)

        # connect reordering operators
        self._op5_2.Input.connect(self._opLabel.Output)
        self._op5_2_cached.Input.connect(self._opLabel.CachedOutput)
        

        # set the final reordering operator's AxisOrder to that of the input
        origOrder = "".join([s for s in self.Input.meta.getTaggedShape()])
        self._op5_2.AxisOrder.setValue(origOrder)
        self._op5_2_cached.AxisOrder.setValue(origOrder)

        # set background values
        self._setBG()
        

    def propagateDirty(self, slot, subindex, roi):
        # just reset the background, that will trigger recomputation
        #FIXME respect roi
        self._setBG()

    ## set the background values of inner operator
    def _setBG(self):
        if self.Background.ready():
            val = self.Background.value
        else:
            val = 0
        bg = np.asarray(val)
        if bg.size == 1:
            bg = np.zeros(self._op5.Output.meta.shape[3:])
            bg[:] = val
        else:
            bg = val
        bg = vigra.taggedView(bg, axistags='ct')
        #bg.flat = [v for (u,v) in np.broadcast(bg, val)]
        bg = bg.withAxes(*'xyzct')
        self._opLabel.Background.setValue(bg)


## parent class for all connected component labeling implementations
class OpLabelingABC(Operator):
    Input = InputSlot()
    Background = InputSlot()

    Output = OutputSlot()
    CachedOutput = OutputSlot()

    def __init__(self, *args, **kwargs):
        super(OpLabelingABC, self).__init__(*args, **kwargs)
        self._cache = None
        self._metaProvider = _OpMetaProvider(parent=self)

    def setupOutputs(self):
        labelType = np.int32

        # remove unneeded old cache
        if self._cache is not None:
            self._cache.Input.disconnect()
            del self._cache

        m = self.Input.meta
        self._metaProvider.setMeta(
            MetaDict({'shape': m.shape, 'dtype': labelType,
                      'axistags': m.axistags}))

        self._cache = OpCompressedCache(parent=self)
        self._cache.name = "OpLabelVolume.OpCompressedCache"
        self._cache.Input.connect(self._metaProvider.Output)
        self.Output.meta.assignFrom(self._cache.Output.meta)
        self.CachedOutput.meta.assignFrom(self._cache.Output.meta)

        s = self.Input.meta.getTaggedShape()
        shape = (s['c'], s['t'])
        self._cached = np.zeros(shape)

        # prepare locks for each channel and time slice
        locks = np.empty(shape, dtype=np.object)
        for c in range(s['c']):
            for t in range(s['t']):
                locks[c, t] = ThreadLock()
        self._locks = locks

    def propagateDirty(self, slot, subindex, roi):
        # a change somewhere makes the whole time-channel-slice dirty
        # (CCL with vigra is a global operation)
        # applies for Background and Input
        for t in range(roi.start[4], roi.stop[4]):
            for c in range(roi.start[3], roi.stop[3]):
                self._cached[c, t] = 0

    def execute(self, slot, subindex, roi, result):
        #FIXME we don't care right now which slot is requested, just return cached CC
        # get the background values
        bg = self.Background[...].wait()
        bg = vigra.taggedView(bg, axistags=self.Background.meta.axistags)
        bg = bg.withAxes(*'ct')

        # do labeling in parallel over channels and time slices
        pool = RequestPool()

        for t in range(roi.start[4], roi.stop[4]):
            for c in range(roi.start[3], roi.stop[3]):
                # only one thread may update the cache for this c and t, other requests
                # must wait until labeling is finished
                self._locks[c, t].acquire()
                if self._cached[c, t]:
                    # this slice is already computed
                    continue
                # update the whole slice
                req = Request(partial(self._updateSlice, c, t, bg[c, t]))
                pool.add(req)

        pool.wait()
        pool.clean()

        req = self._cache.Output.get(roi)
        req.writeInto(result)
        req.block()

        # release locks and set caching flags
        for t in range(roi.start[4], roi.stop[4]):
            for c in range(roi.start[3], roi.stop[3]):
                self._cached [c,t] = 1
                self._locks[c, t].release()

    ## compute the requested slice and put the results into self._cache
    #
    def _updateSlice(self, c, t, bg):
        raise NotImplementedError("This is an abstract method")


## vigra connected components
class _OpLabelVigra(OpLabelingABC):
    def _updateSlice(self, c, t, bg):
        source = vigra.taggedView(self.Input[..., c, t].wait(), axistags='xyzct')
        source = source.withAxes(*'xyz')
        result = vigra.analysis.labelVolumeWithBackground(
            source, background_value=int(bg))
        result = result.withAxes(*'xyzct')

        stop = np.asarray(self.Input.meta.shape)
        start = 0*stop
        start[3:] = (c, t)
        stop[3:] = (c+1, t+1)
        roi = SubRegion(self._cache.Input, start=start, stop=stop)

        self._cache.setInSlot(self._cache.Input, (), roi, result)


## blockedarray connected components
class _OpLabelBlocked(OpLabelingABC):
    def _updateSlice(self, c, t, bg):
        raise NotImplementedError("TODO: implement")

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


