import warnings
from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.operators.adaptors import Op5ifyer
from lazyflow.stype import Opaque
from lazyflow.rtype import SubRegion
import numpy as np
import vigra


np.set_printoptions(linewidth=200)



############################
############################           
############################
###                      ###
###  Top Level Operator  ###
###                      ###
############################
############################
############################



try:
    from ilastik.applets.patchCreator.opPatchCreator import patchify
except ImportError:
    warnings.warn("Could not import ilastik.applets.patchCreator.opPatchCreator (possibly ilastik is not installed)")
    def patchify(img, patchShape, overlap, gridInit, gridShape):
        #FIXME!!
        raise NotImplementedError("No alternative for ilastik patchify function!")

class OpInterpMissingData(Operator):
    name = "OpInterpMissingData"

    InputVolume = InputSlot()
    InputSearchDepth = InputSlot(value=3)
    PatchSize = InputSlot(value=64)
    Output = OutputSlot()
    
    interpolationMethod = 'cubic'
    _requiredMargin = {'cubic': 2, 'linear': 1, 'constant': 0}
    
    def __init__(self, *args, **kwargs):
        super(OpInterpMissingData, self).__init__(*args, **kwargs)
        
        
        self.detector = OpDetectMissing(parent=self)
        self.interpolator = OpInterpolate(parent=self)
        self.interpolator.interpolationMethod = self.interpolationMethod
        
        self.detector.InputVolume.connect(self.InputVolume)
        self.detector.PatchSize.connect(self.PatchSize)
        self.interpolator.InputVolume.connect(self.InputVolume)
        self.interpolator.Missing.connect(self.detector.Output) 


    def setupOutputs(self):
        # Output has the same shape/axes/dtype/drange as input
        self.Output.meta.assignFrom( self.InputVolume.meta )

        # Check for errors
        taggedShape = self.InputVolume.meta.getTaggedShape()
        
        # this assumption is important!
        if 't' in taggedShape:
            assert taggedShape['t'] == 1, "Non-spatial dimensions must be of length 1"
        if 'c' in taggedShape:
            assert taggedShape['c'] == 1, "Non-spatial dimensions must be of length 1"

    def execute(self, slot, subindex, roi, result):
        '''
        execute
        '''
        
        assert self.interpolationMethod in self._requiredMargin.keys(), "Unknown interpolation method {}".format(self.interpolationMethod)
        
        def roi2slice(roi):
            out = []
            for start, stop in zip(roi.start, roi.stop):
                out.append(slice(start, stop))
            return tuple(out)

        
        oldStart = np.asarray([k for k in roi.start])
        oldStop = np.asarray([k for k in roi.stop])

        z_index = self.InputVolume.meta.axistags.index('z')
        nz = self.InputVolume.meta.getTaggedShape()['z']
        
        print("Pre",roi.start)
        # check if more input is needed, and how many
        z_offsets = self._extendRoi(roi)
        print("Offsets", z_offsets)
        print("Post",roi.start)

        # get extended interpolation
        roi.start[z_index] -= z_offsets[0]
        roi.stop[z_index] += z_offsets[1]
        
        a = self.interpolator.Output.get(roi).wait()
        
        # reduce to original roi
        roi.stop = roi.stop - roi.start
        roi.start *= 0
        roi.start[z_index] += z_offsets[0]
        roi.stop[z_index] -= z_offsets[1]
        key = roi2slice(roi)
        
        result[:] = a[key]
        
        
        return result
    
    
    def propagateDirty(self, slot, subindex, roi):
        # TODO: This implementation of propagateDirty() isn't correct.
        #       That's okay for now, since this operator will never be used with input data that becomes dirty.
        #TODO if the input changes we're doing nothing?
        #warnings.warn("FIXME: propagateDirty not implemented!")
        self.Output.setDirty(roi)
        
    
    def _extendRoi(self, roi):
        
        
        origStart = np.copy(roi.start)
        origStop = np.copy(roi.stop)
        
        
        offset_top = 0
        offset_bot = 0
        
        z_index = self.InputVolume.meta.axistags.index('z')
        
        depth = self.InputSearchDepth.value
        nRequestedSlices = roi.stop[z_index] - roi.start[z_index]
        nNeededSlices = self._requiredMargin[self.interpolationMethod]
        
        missing = vigra.taggedView(self.detector.Output.get(roi).wait(), axistags=self.InputVolume.meta.axistags).withAxes(*'zyx')
        
        nGoodSlicesTop = 0
        # go inside the roi
        for k in range(nRequestedSlices):
            if np.all(missing[k,...]==0): #clean slice
                nGoodSlicesTop += 1
            else:
                break
        
        # are we finished yet?
        if nGoodSlicesTop >= nRequestedSlices:
            return (0,0)
        
        # looks like we need more slices on top
        while nGoodSlicesTop < nNeededSlices and offset_top < depth and roi.start[z_index] > 0:
            roi.stop[z_index] = roi.start[z_index] 
            roi.start[z_index] -= 1
            offset_top += 1
            topmissing = self.detector.Output.get(roi).wait()
            if np.all(topmissing==0): # clean slice
                nGoodSlicesTop += 1
            else: # need to start again
                nGoodSlicesTop = 0
        
        
        nGoodSlicesBot = 0
        # go inside the roi
        for k in range(1,nRequestedSlices+1):
            if np.all(missing[-k,...]==0): #clean slice
                nGoodSlicesBot += 1
            else:
                break
        
        
        roi.start = np.copy(origStart)
        roi.stop = np.copy(origStop)
        
        
        # looks like we need more slices on bottom
        while nGoodSlicesBot < nNeededSlices and offset_bot < depth and roi.stop[z_index] < self.InputVolume.meta.getTaggedShape()['z']:
            roi.start[z_index] = roi.stop[z_index] 
            roi.stop[z_index] += 1
            offset_bot += 1
            botmissing = self.detector.Output.get(roi).wait()
            if np.all(botmissing==0): # clean slice
                nGoodSlicesBot += 1
            else: # need to start again
                nGoodSlicesBot = 0
                
        
        roi.start = np.copy(origStart)
        roi.stop = np.copy(origStop)
        
        return (offset_top,offset_bot)
        
        



################################
################################           
################################
###                          ###
###  Interpolation Operator  ###
###                          ###
################################
################################
################################

    
try:
    from scipy.ndimage import label as connectedComponents
except ImportError:
    warnings.warn("Could not import scipy.ndimage.label()")
    def connectedComponents(X):
        #FIXME!!
        return (X,int(X.max()))
    

def _cubic_mat(n=1):
    n = float(n)
    x = -1/(n+1)
    y = (n+2)/(n+1)
    
    A = [[1, x, x**2, x**3],\
        [1, 0, 0, 0],\
        [1, 1, 1, 1],\
        [1, y, y**2, y**3]]
    
    #TODO we could implement the direct inverse here, but it's a bit lengthy
    # compare to http://www.wolframalpha.com/input/?i=invert+matrix%28[1%2Cx%2Cx^2%2Cx^3]%2C[1%2C0%2C0%2C0]%2C[1%2C1%2C1%2C1]%2C[1%2Cy%2Cy^2%2Cy^3]%29
    
    return np.linalg.inv(A)



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
                "InputVolume and Missing must have the same shape ({} vs {})".format(\
                self.InputVolume.meta.getTaggedShape(), self.Missing.meta.getTaggedShape())
            
        

    def execute(self, slot, subindex, roi, result):
        
        # prefill result
        result[:] = self.InputVolume.get(roi).wait()
        
        resultZYXCT = vigra.taggedView(result,self.InputVolume.meta.axistags).withAxes(*'zyxct')
        missingZYXCT = vigra.taggedView(self.Missing.get(roi).wait(),self.Missing.meta.axistags).withAxes(*'zyxct')
        
        (missingLabeled,maxLabel) = connectedComponents(missingZYXCT)
        
        
        for t in range(resultZYXCT.shape[4]):
            for c in range(resultZYXCT.shape[3]):
                for i in range(1,maxLabel+1):
                    self._interpolate(resultZYXCT[...,c,t], missingLabeled[...,c,t]==i)
        
        
        
        return result
        
    def _interpolate(self,volume, missing, method = None):
        '''
        interpolates in z direction
        :param volume: 3d block with axistags 'zyx'
        :type volume: array-like
        :param missing: True where data is missing
        :type missing: bool, 3d block with axistags 'zyx'
        :param method: 'cubic' or 'linear' or 'constant' (see class documentation)
        :type method: str
        '''
        
        method = self.interpolationMethod if method is None else method
        # sanity checks
        assert method in self._requiredMargin.keys(), "Unknown method '{}'".format(method)

        assert volume.axistags.index('z') == 0 \
            and volume.axistags.index('y') == 1 \
            and volume.axistags.index('x') == 2 \
            and len(volume.shape) == 3, \
            "Data must be 3d with z as first axis."
        
        # number and z-location of missing slices (z-axis is at zero)
        #FIXME showstopper, calling where for every patch on the whole image is not wise!
        black_z_ind, black_y_ind, black_x_ind = np.where(missing)
        
        
        if len(black_z_ind) == 0: # no need for interpolation
            return 
        
        # indices with respect to the required margin around the missing values
        minZ = black_z_ind.min() - self._requiredMargin[method]
        maxZ = black_z_ind.max() + self._requiredMargin[method]
        
        
        n = maxZ-minZ-2*self._requiredMargin[method]+1
        
        if not (minZ>-1 and maxZ < volume.shape[0]):
            # this method is not applicable, try another one
            warnings.warn("Margin not big enough for interpolation (need at least {} pixels for '{}')".format(self._requiredMargin[method], method))
            print(missing[:,0,0])
            if self._fallbacks[method] is not None:
                warnings.warn("Falling back to method '{}'".format(self._fallbacks[method]))
                self._interpolate(volume, missing, self._fallbacks[method])
                return
            else:
                assert False, "Margin not big enough for interpolation (need at least {} pixels for '{}') and no fallback available".format(self._requiredMargin[method], method)
                
        minY, maxY = (black_y_ind.min(), black_y_ind.max())
        minX, maxX = (black_x_ind.min(), black_x_ind.max())
        
        if method == 'linear':
            # do a convex combination of the slices to the left and to the right
            xs = np.linspace(0,1,n+2)
            left = volume[minZ,minY:maxY+1,minX:maxX+1]
            right = volume[maxZ,minY:maxY+1,minX:maxX+1]

            for i in range(n):
                # interpolate every slice
                volume[minZ+i+1,minY:maxY+1,minX:maxX+1] =  (1-xs[i+1])*left + xs[i+1]*right
                
        elif method == 'cubic': 
            # interpolation coefficients
            
            D = np.rollaxis(volume[[minZ,minZ+1,maxZ-1,maxZ],minY:maxY+1,minX:maxX+1],0,3)
            F = np.tensordot(D,_cubic_mat(n),([2,],[1,]))
            
            xs = np.linspace(0,1,n+2)
            for i in range(n):
                # interpolate every slice
                x = xs[i+1]
                volume[minZ+i+2,minY:maxY+1,minX:maxX+1] = F[...,0] + F[...,1]*x + F[...,2]*x**2 + F[...,3]*x**3 
                
        else: #constant
            if minZ > 0:
                # fill right hand side with last good slice
                for i in range(maxZ-minZ+1):
                    #TODO what about non-missing values???
                    volume[minZ+i,minY:maxY+1,minX:maxX+1] = volume[minZ-1,minY:maxY+1,minX:maxX+1]
            elif maxZ < volume.shape[0]-1:
                # fill left hand side with last good slice
                for i in range(maxZ-minZ+1):
                    #TODO what about non-missing values???
                    volume[minZ+i,minY:maxY+1,minX:maxX+1] = volume[maxZ+1,minY:maxY+1,minX:maxX+1]
            else:
                # nothing to do for empty block
                warnings.warn("Not enough data for interpolation, leaving slice as is ...")
            
            
############################
############################           
############################
###                      ###
###  Detection Operator  ###
###                      ###
############################
############################
############################

try:
    from sklearn.externals import joblib
    haveScikit = True
except ImportError:
    haveScikit = False


def _histkernel(X,Y):
    res = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            res[i,j] = np.sum(np.where(X[i]<Y[j],X[i],Y[j]))
    
    return res

class SimplestPrediction():
    '''
    predicts 1 for a good patch, 0 for a bad patch
    '''
    
    def fit(self, x=None, y=None):
        pass
    
    def predict(self,X):
        out = np.zeros((X.shape[0],))
        for k, patch in enumerate(X):
            out[k] = not np.all(patch == 0)
            
        return out
    

class OpDetectMissing(Operator):
    InputVolume = InputSlot()
    PatchSize = InputSlot(value=32)
    HaloSize = InputSlot(value=32)
    Output = OutputSlot()
    IsBad = OutputSlot(stype=Opaque)
    
    _detector = SimplestPrediction()
    _outputDtype = np.uint8
    
    
    def __init__(self, *args, **kwargs):
        super(OpDetectMissing, self).__init__(*args, **kwargs)

    def propagateDirty(self, slot, subindex, roi):
        # TODO
        #warnings.warn("FIXME: propagateDirty not implemented!")
        self.Output.setDirty(roi)
    
    def setupOutputs(self):
        self.Output.meta.assignFrom( self.InputVolume.meta )
        self.Output.meta.dtype = self._outputDtype
        self.IsBad.meta.shape = (1,)
        self.IsBad.meta.dtype = np.bool

    def execute(self, slot, subindex, roi, result):
        
        # prefill result
        if slot == self.Output:
            result[:] = 0
            resultZYXCT = vigra.taggedView(result,self.InputVolume.meta.axistags).withAxes(*'zyxct')
        elif slot == self.IsBad:
            resultZYXCT = result
            
        # acquire data
        data = self.InputVolume.get(roi).wait()
        dataZYXCT = vigra.taggedView(data,self.InputVolume.meta.axistags).withAxes(*'zyxct')
        
        
        
        for t in range(dataZYXCT.shape[4]):
            for c in range(dataZYXCT.shape[3]):
                if slot == self.Output:
                    resultZYXCT[...,c,t] = self._detectMissing(slot, dataZYXCT[...,c,t])
                elif slot == self.IsBad and self._detectMissing(slot, dataZYXCT[...,c,t]):
                    return True

        return result
    
    def isMissing(self,data):
        """
        determines if data is missing values or not by use of the _detector attribute
        
        :param data: a (self._patchSize x self._patchSize) slice 
        :type data: array-like
        :returns: bool -- True, if data seems to be missing
        """
        
        newdata = data.view(np.ndarray).reshape((1,-1))
        
        return not self._detector.predict(newdata)[0]
    

        
    def _detectMissing(self, slot, data):
        '''
        detects missing regions and labels each missing region with 
        its own integer value
        :param origData: 3d data 
        :type origData: array-like
        '''
        
        assert data.axistags.index('z') == 0 \
            and data.axistags.index('y') == 1 \
            and data.axistags.index('x') == 2 \
            and len(data.shape) == 3, \
            "Data must be 3d with axis 'zyx'."
        
        result = vigra.VigraArray(data)*0
        
        
        m = self.PatchSize.value
        if m is None or not m>0:
            raise ValueError("PatchSize must be a positive integer")
        
        
        maxZ,maxY,maxX = data.shape
        
        extX = maxX + m - maxX % m if maxX % m != 0 else maxX
        extY = maxY + m - maxY % m if maxY % m != 0 else maxY
        
        
        patchSize = self.PatchSize.value
        haloSize = self.HaloSize.value
        
        if np.any(patchSize>np.asarray(data.shape[1:])):
            warnings.warn("Ignoring small region (shape={})".format(data.shape))
            maxZ=0
            
            
        # walk over slices
        for z in range(maxZ):
            patches, positions = patchify(data[z,:,:].view(np.ndarray), (patchSize, patchSize), (haloSize,haloSize), (0,0), data.shape[1:])
            # walk over patches
            
            for patch, pos in zip(patches, positions):
                if self.isMissing(patch):
                    if slot == self.IsBad:
                        return True
                    else:
                        ystart = pos[0]
                        ystop = min(ystart+patchSize, data.shape[1])
                        xstart = pos[1]
                        xstop = min(xstart+patchSize, data.shape[2])
                        result[z,ystart:ystop,xstart:xstop] = 1
            
            '''
            for y in range(extY//m):
                startY = y*m
                for x in range(extX//m):
                    startX = x*m
                    if self.isMissing(data[z,startY:startY+m,startX:startX+m]):
                        if slot == self.IsBad: # just return that this volume is indeed bad and stop iterating
                            return True
                        else: # label the patch as missing
                            result[z,startY:np.min([startY+m,maxY]),startX:np.min([startX+m,maxX])] = 1
                        
                    else:
                        wasMissing = False
            '''
         
        return False if slot == self.IsBad else result

    
    def _train(self):
        #m = self._patchSize
        m = self.PatchSize.value
        if m is None or not m>0:
            raise ValueError("PatchSize must be a positive integer")
        
        n = 10
        from sklearn import svm
        from sklearn.externals import joblib
        
        goodimages = np.zeros((4*n,m**2))
        goodclass = np.ones((4*n,))
        badimages = np.zeros((n,m**2))
        badclass = np.zeros((n,))
        for i in range(n):
            badimages[i,...] = (np.random.rand(m**2)).astype(np.uint8)
            goodimages[4*i,...] = (np.random.rand(m**2)*5+50).astype(np.uint8)
            goodimages[4*i+1,...] = (np.random.rand(m**2)*5+120).astype(np.uint8)
            goodimages[4*i+2,...] = (np.random.rand(m**2)*5+200).astype(np.uint8)
            goodimages[4*i+3,...] = (np.random.rand(m**2)*1+5).astype(np.uint8)
    
        s = svm.SVC(kernel=_histkernel)
        X = np.concatenate((goodimages,badimages))
        Y = np.concatenate((goodclass,badclass))

        s.fit(X, Y)
        
        self._detector = s
        
        joblib.dump(s, self._dumpedString)
                        
    

if __name__ == "__main__":
    # do a demo of what the software can handle
    raise NotImplementedError("change file locations below.")
    from lazyflow.graph import Graph
    op = OpInterpMissingData(graph=Graph())
    vol = vigra.readHDF5('/home/burger/hci/hci-data/validation_slices_20_40_3200_4000_1200_2000.h5', 'volume/data')
    vol = vol[0:400,...]
    vol = vigra.VigraArray(vol, axistags=vigra.defaultAxistags('xyzc'))
    m = 64
    vol[m*3:m*5,m*3:m*4,5,:] = 0
    vol[m*2:m*4,m*7:m*8,10,:] = 0
    op.InputVolume.setValue(vol)
    res = op.Output[:].wait()
    out = np.concatenate((vol,res))
    vigra.writeHDF5(out, '/home/burger/hci/hci-data/result.h5', 'volume/data')
