from lazyflow.graph import Graph

import numpy as np
import vigra
from lazyflow.operators.opInterpMissingData import OpInterpMissingData

import unittest

class TestInterpMissingData(unittest.TestCase):
    
    def assertArraysAlmostEqual(self,a,b):
        if not np.mean((a-b)**2)<1e-7:
            self.fail("\n" + str(a) + "\n\n != \n\n" + str(b) + "\n")
            
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
        d2=np.ones((10,10,100))
        for i in range(100): d2[:,:,i]*=(i+1)
        d2[:,:,0:10]=0

        d21=np.ones((10,10,100))
        for i in range(100): d21[:,:,i]*=(i+1)
        d21[:,:,0:10]=0

        d22=np.ones((10,10,100))
        for i in range(100): d22[:,:,i]*=(i+1)
        d22[:,:,0:10]=0



        d23=np.ones((10,10,100))
        for i in range(100): d23[:,:,i]*=(i+1)
        d23[:,:,0:10]=0


        #Last layer is empty
        d3=np.ones((10,10,100))
        for i in range(100): d3[:,:,i]*=(i+1)
        d3[:,:,99]=0

        #Second layer is empty
        d4=np.ones((10,10,100))
        for i in range(100): d4[:,:,i]*=(i+1)
        d4[:,:,1]=0

        #First layer is empty
        d5=np.ones((10,10,100))
        for i in range(100): d5[:,:,i]*=(i+1)
        d5[:,:,0]=0

        #Last layer empty
        d6=np.ones((10,10,100))
        for i in range(100): d6[:,:,i]*=(i+1)
        d6[:,:,99]=0

        #all layers are empty
        d7=np.zeros((10,10,100))

        #next to the layer is empty
        d8=np.ones((10,10,100))
        for i in range(100): d8[:,:,i]*=(i+1)
        d8[:,:,98]=0

        # two different linear interpolations
        d9=vigra.taggedView(np.ones((10,10,100)), 'xyz')
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
        self.assertAlmostEqual(res[1,1,0],36)
        self.assertAlmostEqual(res[1,1,-1],45)

        res=op.Output(start = (0,0,30), stop = (10,10,45)).wait()
        assert res[1,1,0]==31
        assert res[1,1,-1]==45

        res=op.Output(start = (0,0,29), stop = (10,10,45)).wait()
        assert res[1,1,0]==30
        assert res[1,1,-1]==45

        res=op.Output(start = (0,0,0), stop = (10,10,45)).wait()
        assert res[1,1,0]==1
        assert res[1,1,-1]==45

        res=op.Output(start = (0,0,0), stop = (10,10,50)).wait()
        assert res[1,1,0]==1
        assert res[1,1,-1]==50

        res=op.Output(start = (0,0,35), stop = (10,10,51)).wait()
        assert res[1,1,0]==36
        assert res[1,1,-1]==51

        res=op.Output(start = (0,0,35), stop = (10,10,70)).wait()
        assert res[1,1,0]==36
        assert res[1,1,-1]==70

        op.InputVolume.setValue( d21 )
        res=op.Output(start = (0,0,0), stop = (10,10,20)).wait()
        assert res[1,1,3]==11

        op.InputVolume.setValue( d22 )
        res=op.Output(start = (0,0,0), stop = (10,10,6)).wait()
        assert res[1,1,0]==11
        assert res[1,1,-1]==11

        op.InputVolume.setValue( d23 )
        res=op.Output(start = (0,0,1), stop = (10,10,2)).wait()
        assert res[1,1,0]==11
        assert res[1,1,-1]==11

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
