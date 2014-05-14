# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# Copyright 2011-2014, the ilastik developers

from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.operators import OpArrayPiper


## base class for testing operators
# pipes the array, implements all operator methods
class _BasicTesting(operator):

    Input = InputSlot()
    Output = OutputSlot()

    def setupOutputs(self):
        self.Output.meta.assignFrom(self.Input.meta)

    def execute(self, slot, subindex, roi, result):
        req = self.input.get(roi)
        req.writeInto(result)
        req.block()

    def setInSlot(self, slot, subindex, roi, value):
        self.Output[roi.toSlice()] = value

    def propagateDirty(self, slot, subindex, roi):
        self.Output.setDirty(roi)



## This class checks the handling of dirty input
# Main use case for this class is in unit tests for other operators:
#
#   class Test(unittest.TestCase):
#       def test(self):
#           op = MyFancyOperator(...)
#           inspector = OpCheckDirtyPropagation(graph=myGraph)
#           inspector.Input.connect(op.Output)
#           ... some operation that should trigger dirtiness ...
#           slots, subindices, rois = inspector.info()
#           ... process info ...
class OpCheckDirtyPropagation(_BasicTesting):

    def __init__(self, *args, **kwargs):
        super(OpCheckDirtyPropagation, self).__init__(*args, **kwargs)
        self.reset()

    def propagateDirty(self, slot, subindex, roi):
        self._rois.append(roi)
        self._inds.append(subindex)
        self._slots.append(slot)
        super(OpCheckDirtyPropagation, self).propagateDirty(
            slot, subindex, roi)

    ## gather information on past calls to propagateDirty()
    # @param reset reset the information (bool)
    # @return a 3-tuple of lists: (slots, subindices, rois)
    def info(self, reset=True):
        s, i, r = self._slots, self._inds, self._rois
        if reset:
            self.reset()
        return s, i, r

    ## reset the gathered information
    def reset(self):
        self._rois = []
        self._inds = []
        self._slots = []


## Track how often an operator calls input slots
class OpTrackExecute(_BasicTesting):
    #Input = InputSlot()
    #Output = OutputSlot()

    def __init__(self, *args, **kwargs):
        super(OpTrackExecute, self).__init__(*args, **kwargs)
        self.reset()
    
    def execute(self, slot, subindex, roi, result):
        self._rois.append(roi)
        self._inds.append(subindex)
        self._slots.append(slot)
        super(OpTrackExecute, self).execute(slot, subindex, roi, result)

    ## gather information on past calls to execute
    # @param reset reset the information (bool)
    # @return a 3-tuple of lists: (slots, subindices, rois)
    def info(self, reset=True):
        s, i, r = self._slots, self._inds, self._rois
        if reset:
            self.reset()
        return s, i, r

    ## reset the gathered information
    def reset(self):
        self._rois = []
        self._inds = []
        self._slots = []

