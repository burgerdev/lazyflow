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

from abc import ABCMeta, abstractmethod

from lazyflow.operator import Operator


class ParallelStrategyABC(object):
    """
    TODO doc
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def map(self, slot, roi, result):
        """
        TODO doc
        """
        raise NotImplementedError


class OpMapParallel(Operator):
    '''
    Decorator for parallelizing regular operators.

    TODO doc
    FIXME currently only one OutputSlot supported
    FIXME opaque output is ignored
    FIXME no constructor args supported
    FIXME higher-level slots not supported
    '''

    def __init__(self, op2decorate, slot2parallelize, strategy,
                 *args, **kwargs):
        '''
        @param op2decorate the class that should be decorated, will be
                           constructed in __init__ (type: subclass of Operator)
        @param slot2parallelize the output slot that should be parallelized
                                (type: str)
        @param strategy a parallelization strategy (type: ParallelStrategyABC)
        '''

        # parent constructor
        super(OpMapParallel, self).__init__(*args, **kwargs)
        self.strategy = strategy
        assert isinstance(strategy, ParallelStrategyABC),\
            "Cannot use this parallelization strategy"

        # sanity checks
        self.s2p = slot2parallelize
        assert self.s2p in map(lambda s: s.name, op2decorate.outputSlots),\
            "Operator {} does not have an output slot{}".format(
                op2decorate.name, self.s2p)

        # yummy copy-pasta

        # replicate input slot definitions
        for innerSlot in sorted(op2decorate.inputSlots,
                                key=lambda s: s._global_slot_id):
            level = innerSlot.level
            outerSlot = innerSlot._getInstance(self, level=level)
            self.inputs[outerSlot.name] = outerSlot
            setattr(self, outerSlot.name, outerSlot)

        # replicate output slot definitions
        for innerSlot in sorted(op2decorate.outputSlots,
                                key=lambda s: s._global_slot_id):
            level = innerSlot.level
            outerSlot = innerSlot._getInstance(self, level=level)
            self.outputs[outerSlot.name] = outerSlot
            setattr(self, outerSlot.name, outerSlot)

        # connect decorated operator
        op = op2decorate(parent=self)

        for k in op.inputs:
            if not k.startswith("_"):
                op.inputs[k].connect(self.inputs[k])
        for k in op.outputs:
            if not (k.startswith("_") or k == self.s2p):
                self.outputs[k].connect(op.outputs[k])
                self.outputs[k].meta.NOTREADY = None
        self._op = op

        # connect dirtyness callback
        def callWhenDirty(slot, roi):
            self.outputs[self.s2p].setDirty(roi)
        self._op.outputs[self.s2p].notifyDirty(callWhenDirty)

    def setupOutputs(self):
        # the slot2parallelize is not connected so we need to set the meta data
        self.outputs[self.s2p].meta.assignFrom(
            self._op.outputs[self.s2p].meta)

    def execute(self, slot, subindex, roi, result):
        # parallelize request
        assert slot == self.outputs[self.s2p], "a slot wasn't wrapped"
        slot = self._op.outputs[self.s2p]
        for i in subindex:
            slot = slot[i]
        self.strategy.map(slot, roi, result)

    def propagateDirty(self, slot, subindex, roi):
        # dirtyness is handled by callback
        pass

    def getWrappedOperator(self):
        return self._op
