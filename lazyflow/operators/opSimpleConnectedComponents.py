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

import numpy as np
import vigra

from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.operators import OpReorderAxes

_LABEL_TYPE = np.uint32
_ALLOWED_INPUTS = set((np.uint8, np.uint32, np.float32))
_INTEGRAL_INPUTS = set((np.uint8, np.uint32))
_FLOAT_INPUTS = set((np.float32,))
assert _ALLOWED_INPUTS == _INTEGRAL_INPUTS | _FLOAT_INPUTS


class OpSimpleConnectedComponents(Operator):
    """
    computes labeling of requested ROI only

    Input is reordered internally, output order is the same as input.
    If multiple time/channel slices are requested at once, the label
    image is computed per time/channel slice (only for spatial axes).

    Dirty propagation just forwards the dirty roi.
    """
    Input = InputSlot()
    Background = InputSlot()

    Output = OutputSlot()

    __Input = OutputSlot()
    __Output = OutputSlot()

    def __init__(self, *args, **kwargs):
        super(OpSimpleConnectedComponents, self).__init__(
            *args, **kwargs)
        self.__opReorderInput = OpReorderAxes(parent=self)
        self.__opReorderInput.Input.connect(self.Input)
        self.__Input.connect(self.__opReorderInput.Output)
        self.__opReorderInput.AxisOrder.setValue('txyzc')

        self.__opReorderOutput = OpReorderAxes(parent=self)
        self.__opReorderOutput.Input.connect(self.__Output)
        self.Output.connect(self.__opReorderOutput.Output)

    def setupOutputs(self):
        dt = self.__Input.meta.dtype
        if dt not in _ALLOWED_INPUTS:
            raise ValueError("Cannot handle dtype {}".format(dt))

        self.__Output.meta.assignFrom(self.__Input.meta)
        self.__Output.meta.dtype = _LABEL_TYPE

        order = self.Input.meta.getTaggedShape().keys()
        self.__opReorderOutput.AxisOrder.setValue(order)

    def execute(self, slot, subindex, roi, result):
        assert slot is self.__Output

        def tag(arr):
            return vigra.taggedView(arr, axistags='txyzc')

        def spatial(arr):
            return arr.withAxes(*'xyz')

        # determine background value
        bg = self.Background.value
        # we might get a numpy 0d array, which we cannot use for vigra's
        # background argument
        if self.__Input.meta.dtype in _FLOAT_INPUTS:
            bg = float(bg)
        elif self.__Input.meta.dtype in _INTEGRAL_INPUTS:
            bg = int(bg)
        else:
            assert False, "Should not get here!"

        result = tag(result)
        data = tag(self.__Input.get(roi).wait())

        for i, t in enumerate(range(roi.start[0], roi.stop[0])):
            for j, c in enumerate(range(roi.start[4], roi.stop[4])):
                result_slice = spatial(result[i:i+1, ..., j:j+1])
                data_slice = spatial(data[i:i+1, ..., j:j+1])

            vigra.analysis.labelMultiArrayWithBackground(
                data_slice, background_value=bg, out=result_slice)

    def propagateDirty(self, slot, subindex, roi):
        self.Output.setDirty(roi)
