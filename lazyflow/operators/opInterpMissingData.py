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
        
        #OLD WARNING warnings.warn("FIXME: This operator should be memory-optimized using request.writeInto()")

        # acquire data
        data = self.InputVolume.get(roi).wait()
        depth= self.InputSearchDepth.value

        data = data.view( vigra.VigraArray )
        data.axistags = self.InputVolume.meta.axistags
        
        
        #TODO if slot == interpolated data
        #TODO if subindex is adequate
        
        missing = self._detectMissing(data.withAxes(*'xyz').transposeToNumpyOrder())
        
        #TODO sanity checks
        #TODO partitioning
        
        #for 'missing rectangles' in missing
        self._interpolate(data.withAxes(*'xyz').transposeToNumpyOrder(),missing)
        
        result[:] = data
        return result
    
        '''
        self._interpMissingLayer(data.withAxes(*'xyz'))

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



        #   apply Interpolation
        if offset0!=0 or offset1!=0:
            self._interpMissingLayer(data)

            #   cut data to origin shape or roi
            if offset0!=0:
                data=data[:,:,offset0:]
            if offset1!=0:
                data=data[:,:,0:-offset1]


        result[:] = data
        return result
        
        '''

    def propagateDirty(self, slot, subindex, roi):
        # TODO: This implementation of propagateDirty() isn't correct.
        #       That's okay for now, since this operator will never be used with input data that becomes dirty.
        #TODO if the input changes we're doing nothing?
        warnings.warn("FIXME: propagateDirty not implemented!")
        self.Output.setDirty(roi)
        
    def _interpolate(self,volume,missing, method = None):
        '''
        interpolates in z direction
        :param volume: 3d block with axistags 'zyx'
        :type volume: array-like
        :param missing: integers greater zero where data is missing
        :type missing: uint8, 3d block with axistags 'zyx'
        :param method: 'cubic' or 'linear'
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
            print(minind,maxind)
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
                

    '''
    def _interpMissingLayer(self, data):
        """
        Description: Interpolates empty layers and stacks of layers in which all values are zero.

        :param data: Must be 3d, in xyz order.
        """
        #TODO determine empty layer by NaNs
        #TODO 
        assert len(data.shape)==3
        
        fl=data.sum(0).sum(0)==0 #False Layer Array

        #Interpolate First Block
        if fl[0]==1:
            for i in range(fl.shape[0]):
                if fl[i]==0:
                    data[:,:,0:i+1]=self._firstBlock(data[:,:,0:i+1])
                    break


        #Interpolate Center Layers
        pos0=0
        pos1=0
        for i in range(1,fl.shape[0]-1):
            if fl[i]==1:
                if fl[i-1]==0:
                    pos0=i-1
                if fl[i+1]==0:
                    pos1=i+1
                if pos1!=0:
                    data[:,:,pos0:pos1+1]=self._centerBlock(data[:,:,pos0:pos1+1])
                pos1=0


        #Interpolate Last Block 
        if fl[fl.shape[0]-1]==1:
            for i in range(fl.shape[0]):
                i_rev=fl.shape[0]-i-1
                if fl[i_rev]==0:
                    data[:,:,i_rev:data.shape[2]]=self._lastBlock(data[:,:,i_rev:data.shape[2]])
                    break

    def _firstBlock(self,data):
        """
        Description: set the values of the first few empty layers to these of the first correct one

                     [first correct layer]
                            |
                     e.g 000764 --> 777764

        :param data: Must be 3d, in xyz order.
        """
        for i in range(data.shape[2]-1):
            data[:,:,i]=data[:,:,data.shape[2]-1]
        return data

    def _centerBlock(self,sub_data):
        """
        Description: interpolates all layers between the first and the last slices

                     e.g 80004 --> 87654

        :param sub_data: Must be 3d, in xyz order.
        """
        sub_data = sub_data.transpose()
        Total=sub_data.shape[0]-1
        L_0=np.array(sub_data[0,:,:], dtype=np.float32)
        L_1=np.array(sub_data[Total,:,:], dtype=np.float32) 
        for t in range(Total+1):
            Layer=(L_1*t+L_0*(Total-t))/Total  
            sub_data[t,:,:]=Layer
        return sub_data.transpose()

    def _lastBlock(self,data):
        """
        Description: set the values of the last few empty layers to these of the last correct one

                     [last correct layer]
                            |
                     e.g. 467000 --> 467777

        :param data: Must be 3d, in xyz order.
        """
        for i in range(data.shape[2]):
            data[:,:,i]=data[:,:,0]
        return data
        
    '''

    def _detectMissing(self, data):
        missing = np.zeros(data.shape, dtype=np.uint8)
        
        for i in range(data.shape[0]):
            missing[i,...] = 1 if self.isMissing(data[i,...]) else 0
        
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