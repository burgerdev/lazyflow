
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

    # PRIVATE ATTRIBUTES
    _blockShape = np.asarray((100, 100, 100))
    _opLabel = None

    def __init__(self, *args, **kwargs):
        super(OpConnectedComponents, self).__init__(*args, **kwargs)

    def setupOutputs(self):
        #self.Output.meta.assignFrom(self.Input.meta)
        #self.Output.meta.dtype = np.uint32
        
        if self._opLabel is not None:
            opLabel = self._opLabel
            self._opLabel = None
            opLabel.BackgroundLabels.disconnect()
            opLabel.Input.disconnect()
            self.Output.disconnect()
            del opLabel
        
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

        self.Output.connect(opLabel.Output)
        self._opLabel = opLabel

    def execute(self, slot, subindex, roi, result):
        assert False, "Shouldn't get here"

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
        shape = [1,1,1]
        blockMax = 250
        inshape = self.Input.meta.getTaggedShape()
        for i, s in enumerate('xyz'):
            if s in inshape:
                n = inshape[s]
                facts = _combine(_factorize(n))
                facts.sort()
                m = 1
                for f in facts:
                    if f < blockMax:
                        m = f
                    else:
                        break
                
                shape[i] = m
        self._blockShape = tuple(shape)


######## BASIC MATH FOR BLOCK SHAPE ########

_primes = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101,    103, 107, 109, 113, 127, 131, 137, 139, 149,
    151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
    233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313,
    317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
    419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499]

def _factorize(n):
    '''
    factorize an integer, return list of prime factors (up to 499)
    '''
    maxP = int(np.sqrt(n))
    for i in range(len(_primes)):
        p = _primes[i]
        if p>maxP:
            return [n]
        if n % p == 0:
            ret = _factorize(n//p)
            ret.append(p)
            return ret
    assert False, "How did you get here???"

def _combine(f):
    '''
    possible combinations of factors of f
    '''
    
    if len(f)<2:
        return f
    ret = []
    for i in range(len(f)):
        n = f.pop(0)
        sub = _combine(f)
        ret += sub
        for s in sub:
            ret.append(s*n)
        f.append(n)
    return ret   
    

