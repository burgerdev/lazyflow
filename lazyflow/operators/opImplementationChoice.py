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


from lazyflow.operator import Operator, InputSlot


class OpImplementationChoice(Operator):
    '''
    Choose from a predefined set of implementations.

    This operator provides a generic way to choose between multiple
    implementations of an operator interface at runtime. The interface
    is passed as a class in the constructor, which implies that dynamic
    slots (like in OperatorWrapper) are not supported. But you can
    write a short operator that has the correct slots.

    The implementationABC has to be a real operator, although it does
    not have to actually do something useful - you just need to inherit
    from operator and provide the necessary slots. See the unit tests
    for examples. Each input slot that is not available for all
    implementations must be optional. All output slots that are not
    available for the current implementation are set unready via the
    meta.NOTREADY flag.

    For each implementation available, only the slots that are
      * available in implementationABC
      * do not start with an underscore (e.g. self._privateSlot)
    are promoted to the wrapper. 

    Note that the implementing classes do not have to be children
    of the implementationABC, they just have to provide the same slots.
    '''

    # the default name of the slot which holds the current choice, can
    # be overridden in constructor
    # (the slot is dynamic to support nested OpImplementationChoice)
    defaultChoiceSlot = "Implementation"

    # fill this with your implementations (type: dict) 
    # (like: {'implementationName': ImplementingOperator}) 
    implementations = {}

    _current_impl = None
    _impl_name = "Unconfigured OpImplementationChoice"
    _custom_name = False
    _op = None

    @property
    def name(self):
        return self._impl_name

    @name.setter
    def name(self, s):
        self._impl_name = s
        self._custom_name = True

    def __init__(self, implementationABC, *args, **kwargs):

        if 'implementations' in kwargs:
            self.implementations = kwargs['implementations']
            del kwargs['implementations']

        if 'choiceSlot' in kwargs:
            choiceSlot = kwargs['choiceSlot']
            assert choiceSlot not in self.inputSlots
            del kwargs['choiceSlot']
        else:
            choiceSlot = self.defaultChoiceSlot
        s = InputSlot(stype='str')
        self.inputs[choiceSlot] = s._getInstance(self)
        self._Implementation = self.inputs[choiceSlot]

        super(OpImplementationChoice, self).__init__(*args, **kwargs)

        # promote API from implementationABC to this operator
        # mostly stolen from OperatorWrapper

        # replicate input slot definitions
        for innerSlot in sorted(implementationABC.inputSlots,
                                key=lambda s: s._global_slot_id):
            if innerSlot.name.startswith("_"):
                continue
            level = innerSlot.level
            outerSlot = innerSlot._getInstance(self, level=level)
            self.inputs[outerSlot.name] = outerSlot
            setattr(self, outerSlot.name, outerSlot)

        # replicate output slot definitions
        for innerSlot in sorted(implementationABC.outputSlots,
                                key=lambda s: s._global_slot_id):
            if innerSlot.name.startswith("_"):
                continue
            level = innerSlot.level
            outerSlot = innerSlot._getInstance(self, level=level)
            self.outputs[outerSlot.name] = outerSlot
            setattr(self, outerSlot.name, outerSlot)

        # set all slots to unready until some implementation provides them
        for k in self.outputs:
            self.outputs[k].meta.NOTREADY = True

    def setupOutputs(self):
        impl = self._Implementation.value
        if impl == self._current_impl:
            return

        if impl not in self.implementations:
            raise ValueError("Implementation '{}' unknown".format(impl))

        # disconnect former implementation
        if self._op is not None:
            op = self._op
            self._op = None
            for k in self.outputs:
                self.outputs[k].meta.NOTREADY = True  # also disconnects
            for k in op.inputs:
                op.inputs[k].disconnect()

        # connect new implementation
        op = self.implementations[impl](parent=self)
        for k in op.inputs:
            if not k.startswith("_") and k in self.inputs:
                op.inputs[k].connect(self.inputs[k])
        for k in op.outputs:
            if not k.startswith("_") and k in self.outputs:
                self.outputs[k].connect(op.outputs[k])
                self.outputs[k].meta.NOTREADY = None
        self._op = op

        self._current_impl = impl
        if not self._custom_name:
            fmt_str = "OpImplementationChoice[selected={}]"
            self._impl_name = fmt_str.format(self._op.name)

    def propagateDirty(self, slot, subindex, roi):
        # dirty propagation is handled by internal operator
        pass

    def setInSlot(self, slot, subindex, key, value):
        # setInSlot is propagated to the connected inner slot
        pass
