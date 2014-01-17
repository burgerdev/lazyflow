
import numpy as np
import vigra

from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.operators import OpLabelImage, OpBlockedConnectedComponents


class OpConnectedComponents(Operator):

    Input = InputSlot()

    # Must be a list: one for each channel of the volume.
    BackgroundLabels = InputSlot(optional=True)

    Output = OutputSlot()

    # class attribute
    useBlocking = False
    _blockShape = np.asarray((100, 100, 100))

    def __init__(self, *args, **kwargs):
        super(OpConnectedComponents, self).__init__(*args, **kwargs)

    def setupOutputs(self):
        self.Output.meta.assignFrom(self.Input.meta)
        self.Output.meta.dtype = np.uint32

    def execute(self, slot, subindex, roi, result):
        if not self._useBlockedVersion():
            opLabel = OpLabelImage(parent=self)
            opLabel.Input.connect(self.Input)
            opLabel.BackgroundLabels.connect(self.BackgroundLabels)
        else:
            self._setBlockShape()
            opLabel = OpBlockedConnectedComponents(parent=self)
            opLabel.BlockShape.setValue(self._blockShape)
            opLabel.Input.connect(self.Input)
            opLabel.BackgroundLabels.connect(self.BackgroundLabels)

        req = opLabel.Output.get(roi)
        req.writeInto(result)
        req.block()  # FIXME neccessary?

    def propagateDirty(self, slot, subindex, roi):
        pass

    ######### PRIVATE METHODS #########

    def _useBlockedVersion(self):
        # blocking enabled ?
        if not OpConnectedComponents.useBlocking:
            return False

        if self.Input.meta.dtype != np.uint8:
            return False

        return True

    def _setBlockShape(self):
        #FIXME implement real block shape
        shape = np.asarray((1, 1, 1))
        inshape = self.Input.meta.getTaggedShape()
        for i, s in enumerate('xyz'):
            if s in inshape:
                shape[i] = inshape[s]
        self._blockShape = shape