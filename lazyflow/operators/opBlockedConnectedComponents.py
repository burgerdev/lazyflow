
from threading import Lock as ThreadLock

import numpy as np
import vigra

from lazyflow.operator import Operator
from lazyflow.slot import InputSlot, OutputSlot
from lazyflow.rtype import SubRegion
from lazyflow.operators import OpCompressedCache

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
class OpBlockedConnectedComponents(Operator):

    Input = InputSlot()

    # If not provided, the entire input is treated as one block
    BlockShape = InputSlot(optional=True)
    Output = OutputSlot()

    def __init__(self, *args, **kwargs):
        super(OpBlockedConnectedComponents, self).__init__(*args, **kwargs)
        if _importFailed:
            raise ImportError("Importing module 'blockedarray' failed:\n"
                              "{}".format(_importMsg))

        pseudo = _PseudoOperator(parent=self)
        pseudo.Input.connect(self.Input)
        self._pseudo = pseudo

        cache = OpCompressedCache(parent=self)
        cache.BlockShape.connect(self.BlockShape)
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
            self._blockshape = np.minimum(self.BlockShape.value,
                                          self.Input.meta.shape)
        else:
            self._blockshape = self.Input.meta.shape
        assert len(self._blockshape) == len(self.Input.meta.shape),\
            "BlockShape is incompatible with Input"

    def execute(self, slot, subindex, roi, result):
        assert slot == self.Output

        # the first thread that requests data will trigger the computation
        self._lock.acquire()
        if not self._dataReady:
            self._computeCC()
            self._dataReady = True
        self._lock.release()

        req = self._cache.Output.get(roi)
        req.writeInto(result)
        req.block()

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
        blockShape = self.BlockShape.value
        source = _Source(self.Input, blockShape)
        sink = _Sink(self._cache)
        cc = blockedarray.dim3.ConnectedComponents(source, blockShape)
        cc.writeToSink(sink)


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
        return self._slot.meta.shape

    def pyReadBlock(self, roi, output):
        assert len(roi) == 2
        roiP = np.asarray(roi[0])
        roiQ = np.asarray(roi[1])
        p = self._p + roiP
        q = p + roiQ - roiP
        if np.any(q > self._q):
            print(self._p, self._q)
            print(roiP, roiQ)
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
