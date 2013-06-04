import logging
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

loggerName = __name__ 
logger = logging.getLogger(loggerName)



try:
    from ilastik.applets.patchCreator.opPatchCreator import patchify
except ImportError:
    logger.warning("Could not import ilastik.applets.patchCreator.opPatchCreator (possibly ilastik is not installed)")
    def patchify(img, patchShape, overlap, gridInit, gridShape):
        #FIXME!!
        raise NotImplementedError("No alternative for ilastik patchify function!")

class OpInterpMissingData(Operator):
    name = "OpInterpMissingData"

    InputVolume = InputSlot()
    InputSearchDepth = InputSlot(value=3)
    PatchSize = InputSlot(value=64)
    HaloSize = InputSlot(value=0)
      
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
        self.detector.HaloSize.connect(self.HaloSize)
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
        print("InterpMissingData execute()")
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
        
        # check if more input is needed, and how many
        z_offsets = self._extendRoi(roi)
        
        
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
        #logger.warning("FIXME: propagateDirty not implemented!")
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
    logger.warning("Could not import scipy.ndimage.label()")
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
        #logger.warning("FIXME: propagateDirty not implemented!")
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
        black_z_ind, black_y_ind, black_x_ind = np.where(missing)
        
        
        if len(black_z_ind) == 0: # no need for interpolation
            return 
        
        # indices with respect to the required margin around the missing values
        minZ = black_z_ind.min() - self._requiredMargin[method]
        maxZ = black_z_ind.max() + self._requiredMargin[method]
        
        
        n = maxZ-minZ-2*self._requiredMargin[method]+1
        
        if not (minZ>-1 and maxZ < volume.shape[0]):
            # this method is not applicable, try another one
            logger.warning("Margin not big enough for interpolation (need at least {} pixels for '{}')".format(self._requiredMargin[method], method))
            if self._fallbacks[method] is not None:
                logger.warning("Falling back to method '{}'".format(self._fallbacks[method]))
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
                logger.warning("Not enough data for interpolation, leaving slice as is ...")
            
            
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
    from sklearn.svm import SVC as MySVM
    #from sklearn.svm import OneClassSVM as MySVM
    #raise ImportError("SVM not available yet.")
except ImportError:
    class MySVM(object):
        
        def __init__(self, kernel=None):
            pass
        
        def fit(self, x=None, y=None):
            pass
        
        def predict(self,X):
            out = np.zeros(len(X))
            for k, patch in enumerate(X):
                out[k] = 0 if np.all(patch[1:] == 0) else 1
            return out


def _histogramIntersectionKernel(X,Y):
    res = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            res[i,j] = np.sum(X[i]+Y[j]-np.abs(X[i]-Y[j]))
    
    return res

def _defaultTrainingSet(defectSize=128):
    vol = vigra.VigraArray(((np.random.rand(200,200,50)-.5)*125+125).astype(np.uint8), axistags=vigra.defaultAxistags('xyz'))
    labels = vigra.VigraArray(np.zeros((200,200,50),dtype=np.uint8), axistags=vigra.defaultAxistags('xyz'))
    
    labels[70:70+defectSize,70:70+defectSize,1:3] = 2
    labels[70:70+defectSize,70:70+defectSize,5:7] = 1
    vol[70:70+defectSize,70:70+defectSize,5:7] = 0
    return (vol, labels)

class OpDetectMissing(Operator):
    InputVolume = InputSlot()
    PatchSize = InputSlot(value=32)
    HaloSize = InputSlot(value=32)
    
    TrainingVolume = InputSlot(value = _defaultTrainingSet()[0])
    
    # labels: {0: unknown, 1: missing, 2: good}
    TrainingLabels = InputSlot(value = _defaultTrainingSet()[1])
    NTrainingSamples = InputSlot(value=250)
    
    Output = OutputSlot()
    #IsBad = OutputSlot()
    
    _detectors = {}
    _outputDtype = np.uint8
    
    
    def __init__(self, *args, **kwargs):
        super(OpDetectMissing, self).__init__(*args, **kwargs)
        
    def propagateDirty(self, slot, subindex, roi):
        # TODO
        #logger.warning("FIXME: propagateDirty not implemented!")
        
        if slot == self.PatchSize or slot == self.HaloSize:
            self._retrain()
        
        self.Output.setDirty(roi)
    
    def setupOutputs(self):
        self.Output.meta.assignFrom( self.InputVolume.meta )
        self.Output.meta.dtype = self._outputDtype
        #self.IsBad.meta.shape = (1,)
        #self.IsBad.meta.dtype = np.bool
        
        tags = self.TrainingVolume.meta.getTaggedShape()
        
        if 't' in tags: assert tags['t'] == 1, "Time axis must be singleton"
        if 'c' in tags: assert tags['c'] == 1, "Channel axis must be singleton"
        
        assert self.TrainingVolume.meta.getTaggedShape() == self.TrainingLabels.meta.getTaggedShape(), \
            "Training labels and training volume must have the same shape."  
        
        
        self._train()

    def execute(self, slot, subindex, roi, result):
        
        # prefill result
        if slot == self.Output:
            result[:] = 0
            resultZYXCT = vigra.taggedView(result,self.InputVolume.meta.axistags).withAxes(*'zyxct')
        #elif slot == self.IsBad:
        #    resultZYXCT = result
            
        # acquire data
        data = self.InputVolume.get(roi).wait()
        dataZYXCT = vigra.taggedView(data,self.InputVolume.meta.axistags).withAxes(*'zyxct')
        
        
        
        for t in range(dataZYXCT.shape[4]):
            for c in range(dataZYXCT.shape[3]):
                if slot == self.Output:
                    resultZYXCT[...,c,t] = self._detectMissing(slot, dataZYXCT[...,c,t])
                #elif slot == self.IsBad and self._detectMissing(slot, dataZYXCT[...,c,t]):
                #    return True

        return result
    
    def isMissing(self,data):
        """
        determines if data is missing values or not 
        
        :param data: a slice
        :type data: array-like
        :returns: bool -- True, if data seems to be missing
        """
        
        if not data.size in self._detectors.keys():
            logger.warning("Encountered invalid patch size ({} not in {}), cannot determine if missing...".format(data.size, self._detectors.keys()))
            return False
        else:
            hist = self._toHistogram(data)
            ans = self._detectors[data.size].predict((hist,))[0]
            return not np.bool(ans)
    

        
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
        
        patchSize = self.PatchSize.value
        haloSize = self.HaloSize.value
        
        if patchSize is None or not patchSize>0:
            raise ValueError("PatchSize must be a positive integer")
        if haloSize is None or not haloSize>=0:
            raise ValueError("HaloSize must be a non-negative integer")
        
        if np.any(patchSize>np.asarray(data.shape[1:])):
            logger.warning("Ignoring small region (shape={})".format(data.shape))
            maxZ=0
        else:
            maxZ = data.shape[0]
            
        # walk over slices
        for z in range(maxZ):
            patches, positions = patchify(data[z,:,:].view(np.ndarray), (patchSize, patchSize), (haloSize,haloSize), (0,0), data.shape[1:])
            # walk over patches
            for patch, pos in zip(patches, positions):
                if self.isMissing(patch):
                    ystart = pos[0]
                    ystop = min(ystart+patchSize, data.shape[1])
                    xstart = pos[1]
                    xstop = min(xstart+patchSize, data.shape[2])
                    result[z,ystart:ystop,xstart:xstop] = 1
         
        return result
        #return False if slot == self.IsBad else result

    def _toHistogram(self,data):
        if self.InputVolume.meta.dtype == np.uint8:
            r = (0,255) 
        elif self.InputVolume.meta.dtype == np.uint16:
            r = (0,65535) 
        else:
            #FIXME
            r = (0,255)
        (hist, _) = np.histogram(data, bins=100, range=r)
        return hist
        
        
    def _train(self, force=False):
        '''
        retrains if patch size is currently untrained or force is True
        '''
        patchSize = self.PatchSize.value + self.HaloSize.value
        if force or not patchSize**2 in self._detectors.keys():
            self._retrain(patchSize)
        
    
    def _retrain(self,  patchSize):
        '''
        trains with samples drawn from slots TrainingVolume and TrainingLabels
        '''
        
        from scipy.signal import convolve2d as conv2d
        
        if not patchSize**2 in self._detectors.keys():
            self._detectors[patchSize**2] = MySVM(kernel=_histogramIntersectionKernel)
        
        vol = vigra.taggedView(self.TrainingVolume[:].wait(),axistags=self.TrainingVolume.meta.axistags).withAxes(*'zyx')
        labels = vigra.taggedView(self.TrainingLabels[:].wait(),axistags=self.TrainingLabels.meta.axistags).withAxes(*'zyx')
        
        def _extractHistograms(vol, labels,  n, borderCases='discard', nPatches=10):
            
            filt = np.ones((patchSize, patchSize))
            out = []
            cond = labels==n 
            
            ind_z, ind_y, ind_x = np.where(cond.view(np.ndarray))
            
            choice = np.random.choice(len(ind_x),size=nPatches, replace=False)
            
            for z,y,x in zip(ind_z[choice], ind_y[choice],ind_x[choice]):
                ymin = y - patchSize//2
                ymax = ymin + patchSize
                xmin = x - patchSize//2
                xmax = xmin + patchSize
                
                if xmin < 0 or ymin < 0 or xmax > vol.shape[2] or ymax > vol.shape[1]:
                    continue

                out.append(self._toHistogram(vol[z,ymin:ymax,xmin:xmax]))

            return out
        
        inliers = _extractHistograms(vol, labels, 1, borderCases='keep', nPatches = self.NTrainingSamples.value)
        outliers = _extractHistograms(vol, labels, 2, borderCases='discard', nPatches = self.NTrainingSamples.value)
        
        
        assert len(inliers)>0 and len(outliers)>0, "Could not extract training data from volume."

        #inliers = inliers[:min(len(inliers),10)]
        #outliers = outliers[:min(len(inliers),10)]
        
        labelIn = [0]*len(inliers)
        labelOut = [1]*len(outliers)
        
        x = inliers+outliers
        y = labelIn+labelOut
        self._detectors[patchSize**2].fit(x, y)
        
        #self._detectors[patchSize**2].fit(inliers)
        
        '''
        print("Result for black slice: ", self.isMissing(vigra.VigraArray(np.zeros((1,patchSize, patchSize), dtype=np.uint8), axistags=vigra.defaultAxistags('zyx'))))
        print("Result for gray slice: ", self.isMissing(vigra.VigraArray(np.ones((1,patchSize, patchSize), dtype=np.uint8)*125, axistags=vigra.defaultAxistags('zyx'))))
        print("Result for white slice: ", self.isMissing(vigra.VigraArray(np.ones((1,patchSize, patchSize), dtype=np.uint8)*255, axistags=vigra.defaultAxistags('zyx'))))
        
        
        print("Support Vectors:")
        print(self._detectors[patchSize**2].support_vectors_)
        '''

if __name__ == "__main__":
    # do a demo of what the software can handle
    raise NotImplementedError("change file locations below.")
    