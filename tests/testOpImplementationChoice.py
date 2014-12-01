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

import numpy as np
import vigra
import unittest

from lazyflow.graph import Graph
from lazyflow.operators import OpImplementationChoice
from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.operators import OpMaxChannelIndicatorOperator
from lazyflow.operators import OpArrayPiper
from lazyflow.operators import OpPixelOperator
from lazyflow.operators import OpSingleChannelSelector

class TestOpImplementationChoice(unittest.TestCase):

    class Base(Operator):
        '''
        Base class for testing implementations, just for code reuse.
        '''
        Input = InputSlot()
        Output = OutputSlot()

        class SetInSlotCalled(Exception):
            '''
            exception for determining whether setInSlot was called
            '''
            pass

        def setupOutputs(self):
            self.Output.meta.assignFrom(self.Input.meta)

        def execute(self, slot, subindex, roi, result):
            req = self.Input.get(roi)
            req.writeInto(result)
            req.block()

        def propagateDirty(self, slot, subindex, roi):
            self.Output.setDirty(roi)

        def setInSlot(self, slot, subindex, key, value):
            raise self.SetInSlotCalled()

    class Ex1(Base):
        '''
        First implementation of Base
        '''
        AltOutput1 = OutputSlot()

        def setupOutputs(self):
            self.Output.meta.assignFrom(self.Input.meta)
            self.AltOutput1.meta.assignFrom(self.Input.meta)

        def propagateDirty(self, slot, subindex, roi):
            self.Output.setDirty(roi)
            self.AltOutput1.setDirty(roi)

    class Ex2(Base):
        '''
        Second implementation of Base
        '''
        AltOutput2 = OutputSlot()

        def setupOutputs(self):
            self.Output.meta.assignFrom(self.Input.meta)
            self.AltOutput2.meta.assignFrom(self.Input.meta)

        def propagateDirty(self, slot, subindex, roi):
            self.Output.setDirty(roi)
            self.AltOutput2.setDirty(roi)

    class ABC(Base):
        '''
        This class is going to be given to OpImplementationChoice, with
        the choices being OpPixelOperator and OpSingleChannelSelector.
        Note: The choice of operators is for demonstration purposes
        only and does not make any programmatic sense.
        '''
        # needed by OpPixelOperator
        Function = InputSlot(optional=True)
        # needed by OpSingleChannelSelector
        Index = InputSlot(optional=True)

    class ABC2(Operator):
        '''
        This class is going to be given to OpImplementationChoice, with
        the choices being Ex1 and Ex2.
        '''
        Input = InputSlot()
        AltOutput1 = OutputSlot()
        AltOutput2 = OutputSlot()
        Output = OutputSlot()

    def setUp(self):
        pass

    def testUsage(self):
        choices = {'pipe': OpArrayPiper,
                   'chan': OpMaxChannelIndicatorOperator}

        wrap = OpImplementationChoice(self.Base, graph=Graph())
        wrap.implementations = choices

        wrap.Implementation.setValue('pipe')
        vol = np.zeros((2, 10, 15, 20, 3))
        vol = vigra.taggedView(vol, axistags='txyzc')
        wrap.Input.setValue(vol)
        # check if connection is done properly
        assert vol.shape == wrap.Output.meta.shape, "inner op not connected"
        # check if piping works
        out = wrap.Output[...].wait()

        vol2 = np.zeros((5, 6, 7))
        vol2 = vigra.taggedView(vol2, axistags='xyz')
        wrap.Input.setValue(vol2)
        # check if setupOutputs still works
        assert vol2.shape == wrap.Output.meta.shape, "setupOutputs not called"

        wrap.Implementation.setValue('chan')
        vol = np.zeros((2, 10, 15, 20, 3))
        vol = vigra.taggedView(vol, axistags='txyzc')
        wrap.Input.setValue(vol)
        # check if operator is switched
        assert wrap.Output.meta.dtype == np.uint8, "op not switched"

    def testAdvancedUsage(self):
        choices = {'pixop': OpPixelOperator,
                   'chan': OpSingleChannelSelector}

        wrap = OpImplementationChoice(self.ABC, graph=Graph())
        wrap.implementations = choices

        wrap.Implementation.setValue('pixop')
        vol = np.zeros((2, 10, 15, 20, 3), dtype=np.int)
        vol = vigra.taggedView(vol, axistags='txyzc')
        wrap.Input.setValue(vol)
        wrap.Function.setValue(lambda x: x+1)
        # check if connection is done properly
        assert vol.shape == wrap.Output.meta.shape, "inner op not connected"
        # check if piping works
        out = wrap.Output[...].wait()
        # check if correct operator in use
        assert np.all(out == 1)

        wrap.Implementation.setValue('chan')
        wrap.Index.setValue(0)
        # check if operator is switched
        out = wrap.Output[...].wait()
        ts = wrap.Output.meta.getTaggedShape()
        if 'c' in ts:
            assert ts['c'] == 1, "op not switched"

    def testVaryingOutputSlots(self):
        choices = {'1': self.Ex1, '2': self.Ex2}

        wrap = OpImplementationChoice(self.ABC2, graph=Graph())
        wrap.implementations = choices

        vol = np.zeros((2, 10, 15, 20, 3), dtype=np.int)
        vol = vigra.taggedView(vol, axistags='txyzc')
        wrap.Input.setValue(vol)

        wrap.Implementation.setValue('1')
        assert wrap.Output.ready()
        assert wrap.AltOutput1.ready()
        assert not wrap.AltOutput2.ready(), str(wrap.AltOutput2)

        wrap.Implementation.setValue('2')
        assert wrap.Output.ready()
        assert not wrap.AltOutput1.ready(), str(wrap.AltOutput1)
        assert wrap.AltOutput2.ready()

    def testSetInSlot(self):
        choices = {'1': self.Ex1, '2': self.Ex2}

        wrap = OpImplementationChoice(self.ABC2, graph=Graph())
        wrap.implementations = choices

        vol = np.zeros((2, 10, 15, 20, 3), dtype=np.int)
        vol = vigra.taggedView(vol, axistags='txyzc')
        wrap.Input.setValue(vol)

        wrap.Implementation.setValue('1')
        with self.assertRaises(self.Base.SetInSlotCalled):
            wrap.Input[0:1, 0:1, 0:1, 0:1, 0:1] = 1

        wrap.Implementation.setValue('2')
        with self.assertRaises(self.Base.SetInSlotCalled):
            wrap.Input[0:1, 0:1, 0:1, 0:1, 0:1] = 1
