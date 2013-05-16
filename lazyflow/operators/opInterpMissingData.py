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
    _fallbacks = {'cubic': 'linear', 'linear': 'constant', 'constant': None}

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
        
        # we want to interpolate in two steps
        #   1. Detection of missing regions
        #   2. Interpolation of each missing region
        
        warnings.warn("FIXME: This operator should be memory-optimized using request.writeInto()")

        
        #TODO if slot == interpolated data
        #TODO if subindex is adequate
        
        # acquire data
        (data,original_slice) = self._getData(roi)
        
        
        # determine missing indices
        missing = self._detectMissing(data.withAxes(*'xyz').transposeToNumpyOrder())
        
        #TODO sanity checks
        
        #TODO what about close missing regions???
        for i in range(1,missing.max()+1):
            newmissing = np.zeros(missing.shape)
            newmissing[missing == i] = 1
            self._interpolate(data.withAxes(*'xyz').transposeToNumpyOrder(),newmissing)
        
        result[:] = data[original_slice]
        return result
    
    
    def propagateDirty(self, slot, subindex, roi):
        # TODO: This implementation of propagateDirty() isn't correct.
        #       That's okay for now, since this operator will never be used with input data that becomes dirty.
        #TODO if the input changes we're doing nothing?
        warnings.warn("FIXME: propagateDirty not implemented!")
        self.Output.setDirty(roi)
        
    def _getData(self, roi):
        data = self.InputVolume.get(roi).wait()
        depth= self.InputSearchDepth.value

        data = data.view( vigra.VigraArray )
        data.axistags = self.InputVolume.meta.axistags
        
        missing = vigra.VigraArray(data, dtype=np.uint8)
        
        z_index = self.InputVolume.meta.axistags.index('z')
        n_layers = self.InputVolume.meta.getTaggedShape()['z']


        old_start = roi.start
        old_stop = roi.stop

        #   while roi top layer is empty, 
        #   push layer from data to top of roi
        offset0=0
        while(self.isMissing(data[:,:,0])):

            #searched depth reached
            if offset0==depth:
                break

            #top layer reached
            if old_start[z_index]-offset0==0:
                break

            offset0+=1
            new_key = (slice(old_start[0], old_stop[0], None), slice(old_start[1], old_stop[1]), \
                    slice(old_start[2]-offset0, old_stop[2]))
            data = self.InputVolume[new_key].wait()
        
        #   while roi bottom layer is empty, 
        #   push layer from data to bottom of roi 
        offset1=0
        while(self.isMissing(data[:,:,-1])):
            #searched depth reached
            if offset1==depth:
                break

            #bottom layer reached
            if old_stop[z_index]+offset1==n_layers:
                break

            offset1+=1
            new_key = (slice(old_start[0], old_stop[0], None), slice(old_start[1], old_stop[1]), \
                    slice(old_start[2]-offset0, old_stop[2]+offset1))
            data = self.InputVolume[new_key].wait()
        
        offsets = np.zeros((2,3))
        for i in range(len(data.shape)):
            offsets[1,i] = data.shape[i]
        offsets[0,z_index] = offset0
        offsets[1,z_index] -= offset1
        
        original_slice = tuple([slice(offsets[0,i],offsets[1,i]) for i in range(offsets.shape[1])])
        
        return (data,original_slice)
        
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
        
        #method = 'linear' if method == 'cubic' else method
        
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
            xs = np.linspace(0,1,n+2)
            left = volume[minind,...]
            right = volume[maxind,...]
            
           
            for i in range(n):
                # interpolate every slice
                volume[minind+i+1,...] = xs[i+1]*right + (1-xs[i+1])*left
                
        elif method == 'cubic': 
            # interpolation coefficients
            F0 = volume[minind,...]
            F1 = volume[minind+1,...]
            F2 = volume[maxind-1,...]
            F3 = volume[maxind,...]
            (A0,A1,A2,A3) = _cubic_coeffs_mat(F0,F1,F2,F3)
            
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
                pass

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
        missing = np.zeros(data.shape, dtype=np.uint8)
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
        return missing
    
    def isMissing(self,data):
        """
        determines if data is missing values or not
        
        :param data: a 2-D slice
        :type data: array-like
        :returns: bool - -True, if data seems to be missing
        """
        #TODO this could be done much better
        return np.sum(data)==0
    
    
    




def _spline(a0, a1, a2, a3, x, x2, x3):
    return a0+a1*x+a2*x2+a3*x3

def _cubic_coeffs(f,g,h,i):
    '''
    cubic interpolation coefficients of vector (x is missing, o is arbitrary)
    oooooofgxxxhioooo
    '''
    # CubicInterpolationMatrix = np.matrix("0, 1, 0, 0;-1, 1, 0, 0; 2, -5, 4, -1;-1, 3, -3, 1")
    #a = g
    #b = -f+g
    #c = 2*f-5*g+4*h-i
    #d = -f+3*g-3*h+i
    
    return (g,g-f,2*f-5*g+4*h-i,-f+3*g-3*h+i)
    
_spline_mat = np.vectorize(_spline, otypes=[np.float])
_cubic_coeffs_mat = np.vectorize(_cubic_coeffs, otypes=[np.float,np.float,np.float,np.float])



if __name__ == "__main__":
    pass