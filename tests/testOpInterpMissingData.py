from lazyflow.graph import Graph

import numpy as np
import vigra
from lazyflow.operators.opInterpMissingData import OpInterpMissingData

import unittest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from scipy.interpolate import UnivariateSpline


np.set_printoptions(precision=3, linewidth=80)

_testDescriptions = ['large block empty', 'single layer empty', 'last layer empty', 'first block empty', \
                    #'second to last layer empty', 'second layer empty', 'first layer empty', \
                    #'multiple blocks empty', 'all layers empty', 'different regions empty', \
                    ]


def _getTestVolume(description, method):
    
    if description == 'large block empty':
        expected_output = _volume(nz=100, method=method)
        volume = vigra.VigraArray(expected_output)
        missing = vigra.VigraArray(expected_output, dtype=np.uint8)
        volume[:,:,30:50] = 0
        missing[:] = 0
        missing[:,:,30:50] = 1
    elif description == 'single layer empty':
        (volume, missing, expected_output) = _singleMissingLayer(layer=30)
    elif description == 'last layer empty':
        (volume, missing, expected_output) = _singleMissingLayer(nz=100, layer=99)
    elif description == 'second to last layer empty':
        (volume, missing, expected_output) = _singleMissingLayer(nz=100, layer=98)
    elif description == 'first layer empty':
        (volume, missing, expected_output) = _singleMissingLayer(nz=100, layer=0)
    elif description == 'second layer empty':
        (volume, missing, expected_output) = _singleMissingLayer(nz=100, layer=1)
    elif description == 'first block empty':
        expected_output = _volume(method=method)
        volume = vigra.VigraArray(expected_output)
        missing = vigra.VigraArray(expected_output, dtype=np.uint8)
        volume[:,:,0:10] = 0
        missing[:] = 0
        missing[:,:,0:10] = 1
    elif description == '':
        pass    
    elif description == '':
        pass
    else:
        raise NotImplementedError("test cube '{}' not available.".format(description))
    
    return (volume, missing, expected_output)

def _volume(nx=10,ny=10,nz=100,method='linear'):
    b = vigra.VigraArray( np.ones((nx,ny,nz)), axistags=vigra.defaultAxistags('xyz') )
    if method == 'linear':
        for i in range(b.shape[2]): b[:,:,i]*=(i+1)
    elif method == 'cubic':
        s = nz/3
        for z in range(b.shape[2]): b[:,:,z]= (z-s)**2*z*250.0/(nz*(nz-s)**2) + 1
    elif method == 'constant':
        b[:] = 124
    else:
        raise NotImplementedError("unknown method '{}'.".format(method))
    
    return b

def _singleMissingLayer(layer=30, nx=10,ny=10,nz=100,method='linear'):
    expected_output = _volume(nx=nx, ny=ny, nz=nz, method=method)
    volume = vigra.VigraArray(expected_output)
    missing = vigra.VigraArray(expected_output, dtype=np.uint8)
    volume[:,:,layer] = 0
    missing[:] = 0
    missing[:,:,layer] = 1 
    return (volume, missing, expected_output)




class TestInterpolation(unittest.TestCase):
    '''
    tests for the interpolation
    '''
    
    
    def setUp(self):
        pass
    
    def testLinearAlgorithm(self):
        pass
    
    def testCubicAlgorithm(self):
        (v,m,orig) = _singleMissingLayer(layer=15, nx=1,ny=1,nz=50,method='cubic')
        v[:,:,10:15] = 0
        m[:,:,10:15] = 1
        interpolationMethod = 'cubic'
        g=Graph()
        op = OpInterpMissingData(graph = g)
        op.InputSearchDepth.setValue(0)
        op.interpolationMethod = interpolationMethod
        op.InputVolume.setValue( v )
        
        # natural comparison
        assert_array_almost_equal(op.Output[:].wait()[:,:,10:15].view(np.ndarray),\
                                orig[:,:,10:15].view(np.ndarray), decimal=3,\
                                err_msg="direct comparison to cubic data")
        
        # scipy spline interpolation
        x = np.zeros(v.shape)
        x[:,:,:] = np.arange(v.shape[2])
        (i,j,k) = np.where(m==0)
        xs = x[i,j,k]
        ys = v.view(np.ndarray)[i,j,k]
        spline = UnivariateSpline(x[:,:,[8, 9, 16, 17]], v[:,:,[8,9,16,17]], k=3, s=0)
        e = spline(np.arange(v.shape[2]))
        
        assert_array_almost_equal(op.Output[:].wait()[:,:,10:15].squeeze().view(np.ndarray),\
                                e[10:15], decimal=3, err_msg="scipy.interpolate.UnivariateSpline comparison")
    


class TestInterpMissingData(unittest.TestCase):
    '''
    tests for the whole detection/interpolation workflow
    '''
    
    
    def setUp(self):
        
        pass
    
    def testLinearBasics(self):
        
        interpolationMethod = 'linear'
        g=Graph()
        op = OpInterpMissingData(graph = g)
        op.InputSearchDepth.setValue(0)
        op.interpolationMethod = interpolationMethod

        for desc in _testDescriptions:
            (volume, _, expected) = _getTestVolume(desc, interpolationMethod)
            op.InputVolume.setValue( volume )
            assert_array_almost_equal(op.Output[:].wait()[:,:,30:50].view(np.ndarray), expected.view(np.ndarray)[:,:,30:50], decimal=2, err_msg="method='{}', test='{}'".format(interpolationMethod, desc))
        
    
    def testCubicBasics(self):
        interpolationMethod = 'cubic'
        g=Graph()
        op = OpInterpMissingData(graph = g)
        op.InputSearchDepth.setValue(0)
        op.interpolationMethod = interpolationMethod

        for desc in _testDescriptions:
            (volume, _, expected) = _getTestVolume(desc, interpolationMethod)
            op.InputVolume.setValue( volume )
            assert_array_almost_equal(op.Output[:].wait()[:,:,30:50].view(np.ndarray), expected.view(np.ndarray)[:,:,30:50], decimal=2, err_msg="method='{}', test='{}'".format(interpolationMethod, desc))
        
    
    def testBoundaryCases(self):
        pass
    
    def testSwappedAxes(self):
        pass
    
    def testRoi(self):
        pass
    
    

class Old(unittest.TestCase):
    
    def assertArraysAlmostEqual(self,a,b):
        assert_array_almost_equal(a,b,decimal=5)
            
    def assertAll(self,b):
        if not np.all(b):
            self.fail("\n" + str(b) + "\n")
            
    def setUp(self):

        #Large block and one single Layer is empty
        d1 = vigra.VigraArray( np.ones((10,10,100)), axistags=vigra.defaultAxistags('xyz') )
        for i in range(100): d1[:,:,i]*=(i+1)
        d1[:,:,30:50]=0
        d1[:,:,70]=0

        #Fist block is empty
        d2=vigra.VigraArray( np.ones((10,10,100)), axistags=vigra.defaultAxistags('xyz') )
        for i in range(100): d2[:,:,i]*=(i+1)
        d2[:,:,0:10]=0

        d21=vigra.VigraArray( np.ones((10,10,100)), axistags=vigra.defaultAxistags('xyz') )
        for i in range(100): d21[:,:,i]*=(i+1)
        d21[:,:,0:10]=0

        d22=vigra.VigraArray( np.ones((10,10,100)), axistags=vigra.defaultAxistags('xyz') )
        for i in range(100): d22[:,:,i]*=(i+1)
        d22[:,:,0:10]=0



        d23=vigra.VigraArray( np.ones((10,10,100)), axistags=vigra.defaultAxistags('xyz') )
        for i in range(100): d23[:,:,i]*=(i+1)
        d23[:,:,0:10]=0


        #Last layer is empty
        d3=vigra.VigraArray( np.ones((10,10,100)), axistags=vigra.defaultAxistags('xyz') )
        for i in range(100): d3[:,:,i]*=(i+1)
        d3[:,:,99]=0

        #Second layer is empty
        d4=vigra.VigraArray( np.ones((10,10,100)), axistags=vigra.defaultAxistags('xyz') )
        for i in range(100): d4[:,:,i]*=(i+1)
        d4[:,:,1]=0

        #First layer is empty
        d5=vigra.VigraArray( np.ones((10,10,100)), axistags=vigra.defaultAxistags('xyz') )
        for i in range(100): d5[:,:,i]*=(i+1)
        d5[:,:,0]=0

        #Last layer empty
        d6=vigra.VigraArray( np.ones((10,10,100)), axistags=vigra.defaultAxistags('xyz') )
        for i in range(100): d6[:,:,i]*=(i+1)
        d6[:,:,99]=0

        #all layers are empty
        d7=np.zeros((10,10,100))

        #next to the layer is empty
        d8=vigra.VigraArray( np.ones((10,10,100)), axistags=vigra.defaultAxistags('xyz') )
        for i in range(100): d8[:,:,i]*=(i+1)
        d8[:,:,98]=0

        # two different linear interpolations
        d9=vigra.VigraArray( np.ones((10,10,100)), axistags=vigra.defaultAxistags('xyz') )
        for i in range(50): d9[:,:,i]*=(i+1)
        for i in range(50): d9[:,:,50+i]*=2*(50+i+1)
        d9[:,:,90] = 0
        d9[:,:,10] = 0
        
        self.d1 = d1
        self.d2 = d2
        self.d21 = d21
        self.d22 = d22
        self.d23 = d23
        self.d3 = d3
        self.d4 = d4
        self.d5 = d5
        self.d6 = d6
        self.d7 = d7
        self.d8 = d8
        self.d9 = d9

    def testBasicLinear(self):

        Ones=np.ones((10,10))
        g=Graph()
        op = OpInterpMissingData(graph = g)
        op.InputVolume.setValue( self.d1 )
        op.InputSearchDepth.setValue(0)
        op.interpolationMethod = 'linear'

        self.assertArraysAlmostEqual(op.Output[:].wait()[:,:,40],Ones*41)
        
        self.assertArraysAlmostEqual(op.Output[:].wait()[:,:,70],Ones*71)
        
        op.InputVolume.setValue( self.d2 )
        self.assertArraysAlmostEqual(op.Output[:].wait()[:,:,4],Ones*11)
        
        op.InputVolume.setValue( self.d3 )
        self.assertArraysAlmostEqual(op.Output[:].wait()[:,:,99],Ones*99)
        
        op.InputVolume.setValue( self.d4 )
        self.assertArraysAlmostEqual(op.Output[:].wait()[:,:,1],Ones*2)
        
        op.InputVolume.setValue( self.d5 )
        self.assertArraysAlmostEqual(op.Output[:].wait()[:,:,0],Ones*2)
        
        op.InputVolume.setValue( self.d6 )
        self.assertArraysAlmostEqual(op.Output[:].wait()[:,:,99],Ones*99)
        
        op.InputVolume.setValue( self.d7 )
        self.assertArraysAlmostEqual(op.Output[:].wait()[:,:,50],Ones*0)
        
        op.InputVolume.setValue( self.d8 )
        self.assertArraysAlmostEqual(op.Output[:].wait()[:,:,98],Ones*99)
        
    def testBasicCubic(self):

        Ones=np.ones((10,10))
        g=Graph()
        op = OpInterpMissingData(graph = g)
        
        op.InputSearchDepth.setValue(0)
        op.interpolationMethod = 'linear'
        
        op.InputVolume.setValue( self.d1 )
        out = op.Output[:].wait()[:,:,40]
        self.assertAll( (out > Ones*38) & (out < Ones*42))
        
    def testMultipleMissingLinear(self):
        Ones=np.ones((10,10))
        g=Graph()
        op = OpInterpMissingData(graph = g)
        
        op.InputSearchDepth.setValue(0)
        op.interpolationMethod = 'linear'
        
        op.InputVolume.setValue( self.d9 )
        out = op.Output[:].wait()[:,:,10]
        self.assertArraysAlmostEqual(out, Ones*11)
        out = op.Output[:].wait()[:,:,90]
        self.assertArraysAlmostEqual(out, Ones*182)


    def testAxesReversedLinear(self):
        d1 = self.d1.transpose()
        d2 = self.d2.transpose()
        d3 = self.d3.transpose()        
        d4 = self.d4.transpose()
        d5 = self.d5.transpose()
        d6 = self.d6.transpose()        
        d7 = self.d7.transpose()        
        d8 = self.d8.transpose()        



        Ones=np.ones((10,10))
        g=Graph()
        op = OpInterpMissingData(graph = g)
        op.InputVolume.setValue( d1 )
        op.InputSearchDepth.setValue(0)
        op.interpolationMethod = 'linear'

        self.assertArraysAlmostEqual(op.Output[:].wait()[40,:,:],Ones*41)
        
        
        self.assertArraysAlmostEqual(op.Output[:].wait()[70,:,:],Ones*71)
        
        op.InputVolume.setValue( d2 )
        self.assertArraysAlmostEqual(op.Output[:].wait()[4,:,:],Ones*11)
        
        op.InputVolume.setValue( d3 )
        self.assertArraysAlmostEqual(op.Output[:].wait()[99,:,:],Ones*99)
        
        op.InputVolume.setValue( d4 )
        self.assertArraysAlmostEqual(op.Output[:].wait()[1,:,:],Ones*2)
        
        op.InputVolume.setValue( d5 )
        self.assertArraysAlmostEqual(op.Output[:].wait()[0,:,:],Ones*2)
        
        op.InputVolume.setValue( d6 )
        self.assertArraysAlmostEqual(op.Output[:].wait()[99,:,:],Ones*99)
        
        op.InputVolume.setValue( d7 )
        self.assertArraysAlmostEqual(op.Output[:].wait()[50,:,:],Ones*0)
        
        op.InputVolume.setValue( d8 )
        self.assertArraysAlmostEqual(op.Output[:].wait()[98,:,:],Ones*99)
        

    def testRoi(self):
        d1 = self.d1
        d21 = self.d21
        d22 = self.d22
        d23 = self.d23

        g=Graph()
        op = OpInterpMissingData(graph = g)
        op.InputVolume.setValue( d1 )
        op.InputSearchDepth.setValue(100)
        op.interpolationMethod = 'linear'

        res=op.Output(start = (0,0,35), stop = (10,10,45)).wait()
        self.assertArraysAlmostEqual(res[1,1,0],36)
        self.assertArraysAlmostEqual(res[1,1,-1],45)
        
        res=op.Output(start = (0,0,30), stop = (10,10,45)).wait()
        self.assertArraysAlmostEqual(res[1,1,0],31)
        self.assertArraysAlmostEqual(res[1,1,-1],45)
        
        res=op.Output(start = (0,0,29), stop = (10,10,45)).wait()
        self.assertArraysAlmostEqual(res[1,1,0],30)
        self.assertArraysAlmostEqual(res[1,1,-1],45)
        
        res=op.Output(start = (0,0,0), stop = (10,10,45)).wait()
        self.assertArraysAlmostEqual(res[1,1,0],1)
        self.assertArraysAlmostEqual(res[1,1,-1],45)
        
        res=op.Output(start = (0,0,0), stop = (10,10,50)).wait()
        self.assertArraysAlmostEqual(res[1,1,0],1)
        self.assertArraysAlmostEqual(res[1,1,-1],50)
        
        res=op.Output(start = (0,0,35), stop = (10,10,51)).wait()
        self.assertArraysAlmostEqual(res[1,1,0],36)
        self.assertArraysAlmostEqual(res[1,1,-1],51)
        
        res=op.Output(start = (0,0,35), stop = (10,10,70)).wait()
        self.assertArraysAlmostEqual(res[1,1,0],36)
        self.assertArraysAlmostEqual(res[1,1,-1],70)

        op.InputVolume.setValue( d21 )
        res=op.Output(start = (0,0,0), stop = (10,10,20)).wait()
        self.assertArraysAlmostEqual(res[1,1,3],11)

        op.InputVolume.setValue( d22 )
        res=op.Output(start = (0,0,0), stop = (10,10,6)).wait()
        self.assertArraysAlmostEqual(res[1,1,0],11)
        self.assertArraysAlmostEqual(res[1,1,-1],11)

        op.InputVolume.setValue( d23 )
        res=op.Output(start = (0,0,1), stop = (10,10,2)).wait()
        self.assertArraysAlmostEqual(res[1,1,0],11)
        self.assertArraysAlmostEqual(res[1,1,-1],11)

    def testdepthsearch(self):
        d1 = self.d1
        g=Graph()
        op = OpInterpMissingData(graph = g)
        op.InputVolume.setValue( d1 )

        op.InputSearchDepth.setValue(0)
        res=op.Output(start = (0,0,32), stop = (10,10,45)).wait()
        assert res[1,1,0]==0
        assert res[1,1,-1]==0

        op.InputSearchDepth.setValue(3)
        res=op.Output(start = (0,0,32), stop = (10,10,40)).wait()
        assert res[1,1,1]==30
        assert res[1,1,-1]==30

        op.InputSearchDepth.setValue(2)
        res=op.Output(start = (0,0,46), stop = (10,10,49)).wait()
        assert res[1,1,1]==51
        assert res[1,1,-1]==51





if __name__ == "__main__":
    import sys
    import nose
    sys.argv.append("--nocapture")    # Don't steal stdout.  Show it on the console as usual.
    sys.argv.append("--nologcapture") # Don't set the logging level to DEBUG.  Leave it alone.
    ret = nose.run(defaultTest=__file__)
    if not ret: sys.exit(1)
    

