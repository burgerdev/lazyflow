
from threading import Lock as ThreadLock
from functools import partial

import numpy as np
import vigra

from lazyflow.operator import Operator
from lazyflow.slot import InputSlot, OutputSlot
from lazyflow.rtype import SubRegion
from lazyflow.request import Request, RequestPool
from lazyflow.operators import OpCompressedCache, OpReorderAxes

# try to import the blockedarray module, fail only if neccessary
try:
    import blockedarray
except ImportError as e:
    _importFailed = True
    _importMsg = str(e)
else:
    _importFailed = False


# This operator takes a (possibly blocked) input image and produces a label
# image in a blockwise fashion to reduce memory consumption. The label image is
# computed once for the whole input volume and stored internally as in-memory
# compressed data.
#
# Caveats:
#   - input must be of dimension 3 and of type uint8  #FIXME
#   - block shape must divide volume shape evenly
class OpBlockedConnectedComponents(Operator):

    Input = InputSlot()

    # Must be a list: one for each channel of the volume. #TODO implement
    BackgroundLabels = InputSlot(optional=True) 

    # Blockshape as array with 3 elements (regardless of number of input axes)
    # If not provided, the volume is blocked heuristically #TODO implement
    BlockShape = InputSlot(optional=True)

    Output = OutputSlot()

    def __init__(self, *args, **kwargs):
        super(OpBlockedConnectedComponents, self).__init__(*args, **kwargs)
        if _importFailed:
            raise ImportError("Importing module 'blockedarray' failed:\n"
                              "{}".format(_importMsg))

        op5 = OpReorderAxes(parent=self)
        op5.Input.connect(self.Input)
        op5.AxisOrder.setValue('xyzct')
        self._op5 = op5

        pseudo = _PseudoOperator(parent=self)
        pseudo.Input.connect(op5.Output)
        self._pseudo = pseudo

        cache = OpCompressedCache(parent=self)
        cache.Input.connect(pseudo.Output)
        self._cache = cache

        self._lock = ThreadLock()
        self._dataReady = False

    def setupOutputs(self):
        assert self.Input.meta.dtype == np.uint8,\
            "Only 3d UInt8 images supported"
        assert len(self.Input.meta.shape) == 3,\
            "Only 3d UInt8 images supported"

        self.Output.meta.assignFrom(self.Input.meta)
        self.Output.meta.dtype = np.uint32

        # Clip blockshape to image bounds
        if self.BlockShape.ready():
            self._blockshape = self.BlockShape.value
        else:
            self._blockshape = self.Input.meta.shape
        assert len(self._blockshape) == len(self.Input.meta.shape),\
            "BlockShape is incompatible with Input"

        # propagate blockShape to cache
        self._cache.BlockShape.setValue(
            np.concatenate(
                (self._blockshape, np.asarray((1, 1)))))

        #FIXME disconnect??
        op5 = OpReorderAxes(parent=self)
        op5.Input.connect(self._cache.Output)
        op5.AxisOrder.setValue(
            "".join([s for s in self.Input.meta.getTaggedShape()]))
        self._op52 = op5

    def execute(self, slot, subindex, roi, result):
        assert slot == self.Output
        roi = roi.copy()

        # the first thread that requests data will trigger the computation
        self._lock.acquire()
        if not self._dataReady:
            self._computeCC()
            self._dataReady = True
        self._lock.release()

        req = self._op52.Output.get(roi)
        req.writeInto(result)
        req.block()
        return result

    def propagateDirty(self, slot, subindex, roi):
        # label image gets invalid if any part is changed
        newRoi = roi.copy()
        newRoi.start = np.asarray(self.Output.meta.shape)*0
        newRoi.stop = np.asarray(self.Output.meta.shape)
        self.Output.setDirty(newRoi)
        self._dataReady = False

    #def setInSlot(self, slot, subindex, roi, value):
        #pass

    def _computeCC(self):
        blockShape = self._blockshape
        pool = RequestPool()
        shape5d = self._op5.Output.meta.shape

        # 3d labeling
        #FIXME missing slicer, emulated by Source/Sink
        def partFun(c, t):
            source = _Source(self._op5.Output, blockShape, c, t)
            sink = _Sink(self._cache, c, t)
            cc = blockedarray.dim3.ConnectedComponents(source, blockShape)
            cc.writeToSink(sink)

        # handle each channel and time slice seperately
        for c in range(shape5d[3]):
            for t in range(shape5d[4]):
                req = Request(partial(partFun, c, t))
                pool.add(req)

        # start labeling in parallel
        pool.wait()
        pool.clean()


class _Source(blockedarray.adapters.SourceABC):
    def __init__(self, slot, blockShape, c, t):
        super(_Source, self).__init__()
        self._slot = slot
        self._blockShape = blockShape
        self._p = np.asarray(slot.meta.shape[0:3], dtype=np.long)*0
        self._q = np.asarray(slot.meta.shape[0:3], dtype=np.long)
        self._c = c
        self._t = t

    def pySetRoi(self, roi):
        assert len(roi) == 2
        self._p = np.asarray(roi[0], dtype=np.long)
        self._q = np.asarray(roi[1], dtype=np.long)

    def pyShape(self):
        return self._slot.meta.shape[0:3]

    def pyReadBlock(self, roi, output):
        assert len(roi) == 2
        roiP = np.asarray(roi[0])
        roiQ = np.asarray(roi[1])
        p = self._p + roiP
        q = p + roiQ - roiP
        if np.any(q > self._q):
            raise IndexError("Requested roi is too large for selected "
                             "roi (previous call to setRoi)")
        p = np.concatenate((p, np.asarray((self._c, self._t))))
        q = np.concatenate((q, np.asarray((self._c+1, self._t+1))))
        sub = SubRegion(self._slot, start=p, stop=q)
        req = self._slot.get(sub)
        req.writeInto(
            vigra.taggedView(output, axistags='xyz').withAxes(*'xyzct'))
        req.block()
        return True


class _Sink(blockedarray.adapters.SinkABC):
    def __init__(self, op, c, t):
        super(_Sink, self).__init__()
        self._op = op
        self._c = c
        self._t = t

    def pyWriteBlock(self, roi, block):
        assert len(roi) == 2
        start = roi[0]+[self._c, self._t]
        stop = roi[1]+[self._c+1, self._t+1]
        sub = SubRegion(self._op.Input, start=start, stop=stop)
        block = vigra.taggedView(block, axistags='xyz').withAxes(*'xyzct')
        self._op.setInSlot(self._op.Input, (), sub, block)


class _PseudoOperator(Operator):
    Input = InputSlot()
    Output = OutputSlot()

    def __init__(self, *args, **kwargs):
        super(_PseudoOperator, self).__init__(*args, **kwargs)

    def setupOutputs(self):
        self.Output.meta.assignFrom(self.Input.meta)
        self.Output.meta.dtype = np.uint32

    def execute(self, slot, subindex, roi, result):
        assert False

    def propagateDirty(self, slot, subindex, roi):
        pass
