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
import threading
import numpy
import vigra
from lazyflow.graph import Graph
from lazyflow.roi import sliceToRoi, roiToSlice
from lazyflow.operators import OpArrayPiper, OpArrayCache
from lazyflow.operators.opArrayCache import has_drtile

class KeyMaker():
    def __getitem__(self, *args):
        return list(*args)
make_key = KeyMaker()

class OpArrayPiperWithAccessCount(OpArrayPiper):
    """
    A simple array piper that counts how many times its execute function has been called.
    """
    def __init__(self, *args, **kwargs):
        super(OpArrayPiperWithAccessCount, self).__init__(*args, **kwargs)
        self.accessCount = 0
        self._lock = threading.Lock()
     
    def execute(self, slot, subindex, roi, result):
        with self._lock:
            self.accessCount += 1        
        super(OpArrayPiperWithAccessCount, self).execute(slot, subindex, roi, result)
         
 
class TestOpArrayCache(object):
 
    def setUp(self):
        self.dataShape = (1,100,100,10,1)
        self.data = (numpy.random.random(self.dataShape) * 100).astype(int)
        self.data = self.data.view(vigra.VigraArray)
        self.data.axistags = vigra.defaultAxistags('txyzc')
 
        graph = Graph()
        opProvider = OpArrayPiperWithAccessCount(graph=graph)
        opProvider.Input.setValue(self.data)
        self.opProvider = opProvider
         
        opCache = OpArrayCache(graph=graph)
        opCache.Input.connect(opProvider.Output)
        opCache.blockShape.setValue( (10,10,10,10,10) )
        opCache.fixAtCurrent.setValue(False)
        self.opCache = opCache
 
    def testCacheAccess(self):
        opCache = self.opCache
        opProvider = self.opProvider        
         
        expectedAccessCount = 0
        assert opProvider.accessCount == expectedAccessCount
         
        # Block-aligned request
        slicing = make_key[0:1, 0:10, 10:20, 0:10, 0:1]
        data = opCache.Output( slicing ).wait()
        data = data.view(vigra.VigraArray)
        data.axistags = opCache.Output.meta.axistags
        expectedAccessCount += 1        
        assert (data == self.data[slicing]).all()
        assert opProvider.accessCount == expectedAccessCount
 
        # Same request should come from cache, so access count is unchanged
        data = opCache.Output( slicing ).wait()
        assert opProvider.accessCount == expectedAccessCount
                 
        # Not block-aligned request
        slicing = make_key[0:1, 5:15, 10:20, 0:10, 0:1]
        data = opCache.Output( slicing ).wait()
        data = data.view(vigra.VigraArray)
        data.axistags = opCache.Output.meta.axistags
        expectedAccessCount += 1
        assert (data == self.data[slicing]).all()
        assert opProvider.accessCount == expectedAccessCount
 
        # Same request should come from cache, so access count is unchanged
        data = opCache.Output( slicing ).wait()
        assert opProvider.accessCount == expectedAccessCount
 
    def testDirtySource(self):
        opCache = self.opCache
        opProvider = self.opProvider        
         
        expectedAccessCount = 0
        assert opProvider.accessCount == expectedAccessCount
 
        # Request
        slicing = make_key[:, 0:50, 15:45, 0:10, :]
        data = opCache.Output( slicing ).wait()
        data = data.view(vigra.VigraArray)
        data.axistags = opCache.Output.meta.axistags
        if has_drtile:
            expectedAccessCount += 1
        else:
            expectedAccessCount += 20            
        assert (data == self.data[slicing]).all()
        assert opProvider.accessCount == expectedAccessCount, \
            "Expected {} accesses, but provider was accessed {} times.".format(expectedAccessCount, opProvider.accessCount)
 
        # Track dirty notifications
        gotDirtyKeys = []
        def handleDirty(slot, roi):
            gotDirtyKeys.append( list(roiToSlice(roi.start, roi.stop)) )
        opCache.Output.notifyDirty(handleDirty)
         
        # Change some of the input data and mark it dirty
        dirtykey = make_key[0:1, 10:20, 20:30, 0:3, 0:1]
        self.data[dirtykey] = 0.12345
        opProvider.Input.setDirty(dirtykey)
         
        assert len(gotDirtyKeys) == 1
        assert gotDirtyKeys[0] == dirtykey
         
        # Same request, but should need to access the data again due to dirtiness
        slicing = make_key[:, 0:50, 15:45, 0:10, :]
        data = opCache.Output( slicing ).wait()
        data = data.view(vigra.VigraArray)
        data.axistags = opCache.Output.meta.axistags
        expectedAccessCount += 1
        assert (data == self.data[slicing]).all()
        assert opProvider.accessCount == expectedAccessCount
         
    def testFixAtCurrent(self):
        opCache = self.opCache
        opProvider = self.opProvider        
 
        opCache.fixAtCurrent.setValue(True)
 
        expectedAccessCount = 0
        assert opProvider.accessCount == expectedAccessCount
        # Request (no access to provider because fixAtCurrent)
        slicing = make_key[:, 0:50, 15:45, 0:1, :]
        data = opCache.Output( slicing ).wait()
        assert opProvider.accessCount == expectedAccessCount
 
        # We haven't accessed this data yet,
        # but fixAtCurrent is True so the cache gives us zeros
        assert (data == 0).all()
 
 
        opCache.fixAtCurrent.setValue(False)
 
        # Request again.  Data should match this time.
        data = opCache.Output( slicing ).wait()
        data = data.view(vigra.VigraArray)
        data.axistags = opCache.Output.meta.axistags
        assert (data == self.data[slicing]).all()
        if has_drtile:
            expectedAccessCount += 1
        else:
            expectedAccessCount += 20            
        assert (data == self.data[slicing]).all()
        assert opProvider.accessCount == expectedAccessCount, \
            "Expected {} accesses, but provider was accessed {} times.".format(expectedAccessCount, opProvider.accessCount)
 
        # Freeze it again
        opCache.fixAtCurrent.setValue(True)
 
        # Track dirty notifications
        gotDirtyKeys = []
        def handleDirty(slot, roi):
            gotDirtyKeys.append( list(roiToSlice(roi.start, roi.stop)) )
        opCache.Output.notifyDirty(handleDirty)
         
        # Change some of the input data and mark it dirty
        dirtykey = make_key[0:1, 15:25, 20:30, 0:3, 0:1]
        self.data[dirtykey] = 0.12345
        opProvider.Input.setDirty(dirtykey)
 
        # Dirtiness not propagated due to fixAtCurrent
        assert len(gotDirtyKeys) == 0
         
        # Same request.  Data should still match the previous data (not yet refreshed)
        data2 = opCache.Output( slicing ).wait()
        data2 = data2.view(vigra.VigraArray)
        data2.axistags = opCache.Output.meta.axistags
        assert opProvider.accessCount == expectedAccessCount
        assert (data2 == data).all()
 
        # Unfreeze.  Previous dirty notifications should now be seen.
        opCache.fixAtCurrent.setValue(False)
        assert len(gotDirtyKeys) == 1
 
        # The dirty notification we got will not exactly match the dirty data (it will be block-aligned),
        # but it should be a superset of the real dirty data
        expectedroi = sliceToRoi(dirtykey, opProvider.Output.meta.shape)
        receivedroi = sliceToRoi(gotDirtyKeys[0], opProvider.Output.meta.shape)
        assert (receivedroi[0] <= expectedroi[0]).all()
        assert (receivedroi[1] >= expectedroi[1]).all()
         
        # Same request.  Data should be updated now that we're unfrozen.
        data = opCache.Output( slicing ).wait()
        data = data.view(vigra.VigraArray)
        data.axistags = opCache.Output.meta.axistags
        assert (data == self.data[slicing]).all()
        if has_drtile:
            expectedAccessCount += 1
        else:
            expectedAccessCount += 2            
        assert opProvider.accessCount == expectedAccessCount, \
            "Expected {} accesses, but provider was accessed {} times.".format(expectedAccessCount, opProvider.accessCount)
 
        #### Repeat plain dirty test to ensure fixAtCurrent didn't mess up the block states.
 
        gotDirtyKeys = []
 
        # Change some of the input data and mark it dirty
        dirtykey = make_key[0:1, 10:20, 20:30, 0:3, 0:1]
        self.data[dirtykey] = 0.54321
        opProvider.Input.setDirty(dirtykey)
 
        assert len(gotDirtyKeys) == 1
        assert gotDirtyKeys[0] == dirtykey
         
        # Should need access again.
        slicing = make_key[:, 0:50, 15:45, 0:10, :]
        data = opCache.Output( slicing ).wait()
        data = data.view(vigra.VigraArray)
        data.axistags = opCache.Output.meta.axistags
        expectedAccessCount += 1
        assert (data == self.data[slicing]).all()
        assert opProvider.accessCount == expectedAccessCount
 
    def testUncachedBehaviour(self):
        opCache = self.opCache
        opProvider = self.opProvider        
 
        opCache.fixAtCurrent.setValue(True)
         
        # Track dirty notifications
        gotDirtyKeys = []
        def handleDirty(slot, roi):
            gotDirtyKeys.append( list(roiToSlice(roi.start, roi.stop)) )
        opCache.Output.notifyDirty(handleDirty)
         
        dirtykey = make_key[0:1, 15:25, 20:30, 0:3, 0:1]
        self.data[dirtykey] = 0.12345
        opProvider.Input.setDirty(dirtykey)
 
        # Dirtiness not propagated due to fixAtCurrent
        assert len(gotDirtyKeys) == 0
 
        # Unfreeze
        opCache.fixAtCurrent.setValue(False)
 
        # Output should be dirty from previous change
        assert len(gotDirtyKeys) == 1, \
            "Expected 1 dirty notification, got {}".format( len(gotDirtyKeys) )

    def testCleanBlocksSlot(self):
        self.testCacheAccess()
        opCache = self.opCache
        clean_block_rois = opCache.CleanBlocks.value
        assert [[0, 0, 10, 0, 0], [1, 10, 20, 10, 1]] in clean_block_rois
        assert [[0, 10, 10, 0, 0], [1, 20, 20, 10, 1]] in clean_block_rois
         
 
class TestOpArrayCacheWithObjectDtype(object):
    """
    This test is here to convince me that the OpArrayCache can be used with objects as the dtype.
    (Check whether or not the implementation relies on any operations that are not supported for arrays of dtype=object)
    """
    def test(self):
        class SpecialNumber(object):
            def __init__(self, x):
                self.n = x
         
        data = numpy.ndarray(shape=(2,3), dtype=object)
        data = data.view(vigra.VigraArray)
        data.axistags = vigra.defaultAxistags('tc')
        for i in range(2):
            for j in range(3):
                data[i,j] = SpecialNumber(i*j)
 
        graph = Graph()
        op = OpArrayCache(graph=graph)
        op.Input.setValue(data)
        op.blockShape.setValue( (1,3) )
        assert op.Output.meta.shape == (2,3)
        outputData = op.Output[:].wait()
         
        # Can't use (outputData == data).all() here because vigra doesn't do the right thing if dtype is object.
        for x,y in zip(outputData.flat, data.flat):
            assert x == y
        
class TestOpArrayCache_setInSlot(object):
    
    def test(self):
        """
        Test use-case from https://github.com/ilastik/lazyflow/issues/111
        """
        data = numpy.zeros((20,20))
        data = vigra.taggedView(data, 'xy')
        op = OpArrayCache(graph=Graph())
        op.Input.setValue(data)
        
        # Should not crash...
        op.Input[0:20,0:20] = numpy.ones((20,20))


if __name__ == "__main__":
    import sys
    import nose
    sys.argv.append("--nocapture")    # Don't steal stdout.  Show it on the console as usual.
    sys.argv.append("--nologcapture") # Don't set the logging level to DEBUG.  Leave it alone.
    ret = nose.run(defaultTest=__file__)
    if not ret: sys.exit(1)
