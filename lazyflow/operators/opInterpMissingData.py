import warnings
from lazyflow.graph import Operator, InputSlot, OutputSlot
import numpy as np
import vigra



class OpInterpMissingData(Operator):
    name = "OpInterpMissingData"

    InputVolume = InputSlot()
    InputSearchDepth = InputSlot()
    Output = OutputSlot()
    
    interpolationMethod = 'cubic'
    _requiredMargin = {'cubic': 2, 'linear': 1, 'constant': 0}
    
    def __init__(self, *args, **kwargs):
        super(OpInterpMissingData, self).__init__(*args, **kwargs)
        
        self.detector = OpDetectMissing(parent=self)
        self.interpolator = OpInterpolate(parent=self)
        self.interpolator.interpolationMethod = self.interpolationMethod
        
        self.detector.InputVolume.connect(self.InputVolume)
        self.detector.InputSearchDepth.connect(self.InputSearchDepth)
        self.interpolator.InputVolume.connect(self.InputVolume)
        self.interpolator.Missing.connect(self.detector.Output)
        
        self.InputSearchDepth.setValue(0)
        


    def setupOutputs(self):
        # Output has the same shape/axes/dtype/drange as input
        self.Output.meta.assignFrom( self.InputVolume.meta )

        # Check for errors
        taggedShape = self.InputVolume.meta.getTaggedShape()

        if 't' in taggedShape:
            assert taggedShape['t'] == 1, "Non-spatial dimensions must be of length 1"
        if 'c' in taggedShape:
            assert taggedShape['c'] == 1, "Non-spatial dimensions must be of length 1"

    def execute(self, slot, subindex, roi, result):
        '''
        execute
        '''
        
        '''
        # FUTURE:
        a = self.detector.Output.get(roi).wait()
        while isMissing(a[rim]) or stopIterating:
            roi = largerRoi
            a = self.detector.Output.get(roi).wait()
        
        a = self.interpolator.Output.get(roi).wait()
        result[:] = a[oldRoi]
        '''
        result[:] = self.interpolator.Output.get(roi).wait()
        return result
    
    
    def propagateDirty(self, slot, subindex, roi):
        # TODO: This implementation of propagateDirty() isn't correct.
        #       That's okay for now, since this operator will never be used with input data that becomes dirty.
        #TODO if the input changes we're doing nothing?
        #warnings.warn("FIXME: propagateDirty not implemented!")
        self.Output.setDirty(roi)
        



##############################
### Interpolation Operator ###
##############################


def _spline(a0, a1, a2, a3, x, x2, x3):
    return a0+a1*x+a2*x2+a3*x3

def _cubic_coeffs(f,g,h,i,n=1):
    '''
    cubic interpolation coefficients of vector (x is missing, o is arbitrary)
    oooooofgxxxhioooo
            |||
             n times
    '''
    
    n = float(n)
    '''
    dg = (g-f)/(n+1)
    dh = (i-h)/(n+1)
    
    return (g,dg,-3*g-2*dg+3*h-dh,2*g+dg-2*h+dh)
    '''
    # more natural approach:
    x = [-1/(n+1), 0, 1, (n+2)/(n+1)]
    A = np.fliplr(np.vander(x))
    y = np.linalg.solve(A,[f,g,h,i])
    return (y[0],y[1],y[2],y[3])
    
    
_spline_mat = np.vectorize(_spline, otypes=[np.float])
_cubic_coeffs_mat = np.vectorize(_cubic_coeffs, otypes=[np.float,np.float,np.float,np.float])


class OpInterpolate(Operator):
    InputVolume = InputSlot()
    Missing = InputSlot()
    Output = OutputSlot()
    
    interpolationMethod = 'cubic'
    _requiredMargin = {'cubic': 2, 'linear': 1, 'constant': 0}
    _fallbacks = {'cubic': 'linear', 'linear': 'constant', 'constant': None}
    
    def propagateDirty(self, slot, subindex, roi):
        # TODO
        #warnings.warn("FIXME: propagateDirty not implemented!")
        self.Output.setDirty(roi)
    
    def setupOutputs(self):
        # Output has the same shape/axes/dtype/drange as input
        self.Output.meta.assignFrom( self.InputVolume.meta )

        assert self.InputVolume.meta.getTaggedShape() == self.Missing.meta.getTaggedShape(), \
                "InputVolume and Missing must have the same shape"
            
        # Check for errors
        taggedShape = self.InputVolume.meta.getTaggedShape()
        if 't' in taggedShape:
            assert taggedShape['t'] == 1, "Non-spatial dimensions must be of length 1"
        if 'c' in taggedShape:
            assert taggedShape['c'] == 1, "Non-spatial dimensions must be of length 1"
        

    def execute(self, slot, subindex, roi, result):
        data = self.InputVolume.get(roi).wait()
        data = vigra.VigraArray(data, axistags=self.InputVolume.meta.axistags)
        missing = self.Missing.get(roi).wait()
        #TODO what about close missing regions???
        for i in range(1,missing.max()+1):
            newmissing = vigra.VigraArray(np.zeros(missing.shape), axistags=self.InputVolume.meta.axistags)
            newmissing[missing == i] = 1
            self._interpolate(data.withAxes(*'zyx'),newmissing.withAxes(*'zyx'))
        
        result[:] = data
        return result
        
    def _interpolate(self,volume,missing, method = None):
        '''
        interpolates in z direction
        :param volume: 3d block with axistags 'zyx'
        :type volume: array-like
        :param missing: integers greater zero where data is missing
        :type missing: uint8, 3d block with axistags 'zyx'
        :param method: 'cubic' or 'linear' or 'constant' (see class documentation)
        :type method: str
        '''
        
        method = self.interpolationMethod if method is None else method
        # sanity checks
        assert method in self._requiredMargin.keys(), "Unknown method '{}'".format(method)
        
        # number and z-location of missing slices (z-axis is at zero)
        black_z_ind = np.where(missing>0)[0]
        
        if len(black_z_ind) == 0: # no need for interpolation
            return 
        
        # indices with respect to the required margin around the missing values
        minind = black_z_ind.min() - self._requiredMargin[method]
        maxind = black_z_ind.max() + self._requiredMargin[method]
        
        n = maxind-minind-2*self._requiredMargin[method]+1
        
        if not (minind>-1 and maxind < volume.shape[0]):
            # this method is not applicable, try another one
            warnings.warn("Margin not big enough for interpolation (need at least {} pixels for '{}')".format(self._requiredMargin[method], method))
            if self._fallbacks[method] is not None:
                warnings.warn("Falling back to method '{}'".format(self._fallbacks[method]))
                self._interpolate(volume, missing, self._fallbacks[method])
                return
            else:
                assert False, "Margin not big enough for interpolation (need at least {} pixels for '{}') and no fallback available".format(self._requiredMargin[method], method)
        
        if method == 'linear':
            # do a convex combination of the slices to the left and to the right
            xs = np.linspace(0,1,n+2)
            left = volume[minind,...]
            right = volume[maxind,...]

            for i in range(n):
                # interpolate every slice
                volume[minind+i+1,...] =  (1-xs[i+1])*left + xs[i+1]*right
                
        elif method == 'cubic': 
            # interpolation coefficients
            F0 = volume[minind,...]
            F1 = volume[minind+1,...]
            F2 = volume[maxind-1,...]
            F3 = volume[maxind,...]
            (A0,A1,A2,A3) = _cubic_coeffs_mat(F0,F1,F2,F3,n)
            
            xs = np.linspace(0,1,n+2)
            for i in range(n):
                # interpolate every slice
                x = xs[i+1]
                volume[minind+i+2,...] = _spline_mat(A0, A1, A2, A3, x, x**2, x**3)
                
        else: #constant
            if minind > 0:
                # fill right hand side with last good slice
                for i in range(maxind-minind+1):
                    #TODO what about non-missing values???
                    volume[minind+i,...] = volume[minind-1,...]
            elif maxind < volume.shape[0]-1:
                # fill left hand side with last good slice
                for i in range(maxind-minind+1):
                    #TODO what about non-missing values???
                    volume[minind+i,...] = volume[maxind+1,...]
            else:
                # nothing to do for empty block
                warnings.warn("Not enough data for interpolation, leaving slice as is ...")
            
            
##########################
### Detection Operator ###
##########################
    
class OpDetectMissing(Operator):
    InputVolume = InputSlot()
    InputSearchDepth = InputSlot()
    Output = OutputSlot()
    
    def __init__(self, *args, **kwargs):
        super(OpDetectMissing, self).__init__(*args, **kwargs)
    
    def propagateDirty(self, slot, subindex, roi):
        # TODO
        #warnings.warn("FIXME: propagateDirty not implemented!")
        self.Output.setDirty(roi)
    
    def setupOutputs(self):
        self.Output.meta.assignFrom( self.InputVolume.meta )
        self.Output.meta.dtype = np.uint32


    def execute(self, slot, subindex, roi, result):
        
        # acquire data
        data = self.InputVolume.get(roi).wait()
        
        data = vigra.VigraArray(data, axistags=self.InputVolume.meta.axistags)
        
        # determine missing indices
        missing = self._detectMissing(data)
        
        result[:] = missing
        return result
        
    def _detectMissing(self, data):
        '''
        detects missing regions and labels each missing region with 
        its own integer value
        :param data: 3d data with z as first axis
        :type data: array-like
        :returns: 3d integer block with non-zero integers for missing values
        
        TODO This method just marks whole slices as missing, should be done for 
        smaller regions, too
        '''
        
        result = vigra.VigraArray(np.zeros(data.shape, dtype=np.uint8), axistags=self.InputVolume.meta.axistags)
        missing = result.transposeToNumpyOrder()
        data = data.transposeToNumpyOrder()
        missingInt = 0
        wasMissing = False
        for i in range(data.shape[0]):
            if self.isMissing(data[i,...]):
                if not wasMissing:
                    missingInt += 1
                wasMissing = True
                missing[i,...] = missingInt 
            else:
                wasMissing = False
        return result
    
    def isMissing(self,data):
        """
        determines if data is missing values or not
        
        :param data: a 2-D slice
        :type data: array-like
        :returns: bool - -True, if data seems to be missing
        """
        #TODO this could be done much better
        return np.sum(data)==0

if __name__ == "__main__":
    # do a demo of what the software can handle
    from lazyflow.graph import Graph
    op = OpInterpMissingData(graph=Graph())
    vol = vigra.readHDF5('/home/markus/Coding/hci/hci-data/missingslices.h5', 'volume/data')
    vol = vigra.VigraArray(vol, axistags=vigra.defaultAxistags('xyzc'))
    op.InputVolume.setValue(vol)
    res = op.Output[:].wait()
    vigra.writeHDF5(res, '/home/markus/Coding/hci/hci-data/filledslices.h5', 'volume/data')
