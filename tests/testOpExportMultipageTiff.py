import os
import sys
import shutil
import tempfile

import numpy
import vigra

import lazyflow.graph
from lazyflow.operators.opReorderAxes import OpReorderAxes
from lazyflow.operators.operators import OpArrayCache
from lazyflow.operators.ioOperators import OpExportMultipageTiff, OpInputDataReader

import logging
logger = logging.getLogger('tests.testOpMultipageTiff')

class TestOpMultipageTiff(object):

    def setUp(self):
        self.graph = lazyflow.graph.Graph()
        self._tmpdir = tempfile.mkdtemp()

        # Generate some test data
        self.dataShape = (1, 10, 64, 128, 2)
        self._axisorder = 'tzyxc'
        self.testData = vigra.VigraArray( self.dataShape,
                                         axistags=vigra.defaultAxistags(self._axisorder),
                                         order='C' ).astype(numpy.uint8)
        self.testData[...] = numpy.indices(self.dataShape).sum(0)

    def tearDown(self):
        # Clean up
        shutil.rmtree(self._tmpdir)
        
    def test_basic(self):
        opData = OpArrayCache( graph=self.graph )
        opData.blockShape.setValue( self.testData.shape )
        opData.Input.setValue( self.testData )
        
        filepath = os.path.join( self._tmpdir, 'multipage.tiff' )
        logger.debug( "writing to: {}".format(filepath) )
        
        opExport = OpExportMultipageTiff(graph=self.graph)
        opExport.Filepath.setValue( filepath )
        opExport.Input.connect( opData.Output )

        # Run the export
        opExport.run_export()

        opReader = OpInputDataReader( graph=self.graph )
        opReader.FilePath.setValue( filepath )

        # The reader assumes xyzc order.
        # We have to transpose the data before we compare.
        opReorderAxes = OpReorderAxes( graph=self.graph )
        opReorderAxes.AxisOrder.setValue( self._axisorder )
        opReorderAxes.Input.connect( opReader.Output )
        
        readData = opReorderAxes.Output[:].wait()
        logger.debug("Expected shape={}".format( self.testData.shape ) )
        logger.debug("Read shape={}".format( readData.shape ) )
        
        assert opReorderAxes.Output.meta.shape == self.testData.shape, "Exported files were of the wrong shape or number."
        assert (opReorderAxes.Output[:].wait() == self.testData.view( numpy.ndarray )).all(), "Exported data was not correct"

if __name__ == "__main__":
    # Run this file independently to see debug output.
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    ioOpLogger = logging.getLogger('lazyflow.operators.ioOperators')
    ioOpLogger.addHandler( logging.StreamHandler(sys.stdout) )
    ioOpLogger.setLevel(logging.DEBUG)

    import sys
    import nose
    sys.argv.append("--nocapture")    # Don't steal stdout.  Show it on the console as usual.
    sys.argv.append("--nologcapture") # Don't set the logging level to DEBUG.  Leave it alone.
    nose.run(defaultTest=__file__)
