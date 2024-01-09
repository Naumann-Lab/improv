import numpy as np
import time
import pyqtgraph
from pyqtgraph import EllipseROI, PolyLineROI, ColorMap
from PyQt5 import QtGui,QtCore,QtWidgets
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMessageBox, QApplication
from matplotlib.colors import ListedColormap

from improv.actor import Signal
from . import video_photostim

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# how can i make the UI file to be chosen in the yaml file
class FrontEnd(QtWidgets.QMainWindow, video_photostim_paramset.Ui_MainWindow):

    COLOR = {0: ( 240, 122,  5),
             1: (181, 240,  5),
             2: (5, 240,  5),
             3: (5,  240,  181),
             4: (5,  122, 240),
             5: (64,  5, 240),
             6: ( 240,  5, 240),
             7: ( 240, 5, 64),
             8: ( 240, 240, 240)}

    def __init__(self, visual, comm, parent=None):
        ''' Setup GUI
            Setup and start Nexus controls
        '''
        self.visual = visual #Visual class that provides plots and images
        self.comm = comm #Link back to Nexus for transmitting signals

        self.total_times = []
        self.first = True
        self.prev = 0

        # setting up the gui UI file
        pyqtgraph.setConfigOption('background', QColor(100, 100, 100))
        super(FrontEnd, self).__init__(parent)
        self.setupUi(self)
        pyqtgraph.setConfigOptions(leftButtonPan=False)

        # initiatlize plots (i.e. line graphs)
        self.customizePlots()

        # functions for each button
        self.pushButton_3.clicked.connect(_call(self._runProcess)) #Tell Nexus to start
        self.pushButton_3.clicked.connect(_call(self.update)) #Update front-end graphics
        self.pushButton_2.clicked.connect(_call(self._setup))
        self.gophotostimulate.clicked.connect(_call(self.goPhotostim)) # send signal to photostimulate
        self.checkBox.stateChanged.connect(self.update) #Show live front-end updates

        # placing gui on desktop screen
        topLeftPoint = QApplication.desktop().availableGeometry().topLeft()
        self.move(topLeftPoint)

    def goPhotostim(self):
        # converting the gui values into a params dict to be sent to the photostim actor
        params_dict = {}
        
        ### TODO: (eventually) make this flexible for number and z plane
        params_dict['procedure'] = 'galvo'
        if self.useSLM.isChecked(): #use the slm if the Use SLM box is checked
            params_dict['procedure'] = 'slm-2d'

        params_dict['durations'] = [float(self.duration_val.text())]
        params_dict['powers'] = [float(self.laserpower_val.text())]
        params_dict['revolutions'] = [float(self.spiralrevolutions_val.text())]
        params_dict['sizes'] = [float(self.spiralsize_val.text())]
        params_dict['delays'] = [float(self.interpointdelay_val.text())] * neur_number
        params_dict['repetitions'] = [float(self.repetition_val.text())] 
        params_dict['iterations'] = [float(self.iteration_val.text())]
        params_dict['iteration_delay'] = [float(self.iterationdelay_val.text())] 

        self.visual.q_params_dict.put([params_dict])
        logger.info('{}'.format('sent photostim parameters dict'))

    def update(self):
        ''' Update visualization while running
        '''
        t = time.time()
        self.visual.getData()
        self.selected_neuron = 0

        if self.draw:
            try:
                self.updateLines()
            except Exception as e:
                print('update lines error {}'.format(e))
                import traceback
                print('---------------------Exception in update lines: ' , traceback.format_exc())
            try:
                self.updateVideo()
            except Exception as e:
                logger.error('Error in FrontEnd update Video:  {}'.format(e))
                import traceback
                print('---------------------Exception in update video: ' , traceback.format_exc())

        if self.checkBox.isChecked():
            self.draw = True
        else:
            self.draw = False    
        self.visual.draw = self.draw
            
        QtCore.QTimer.singleShot(10, self.update)
        
        self.total_times.append([self.visual.frame_num, time.time()-t])

    def customizePlots(self):
        self.checkBox.setChecked(True)
        self.draw = True

        #init line plot
        self.flag = True # keyword for circles to be drawn
        self.flagW = True
        self.flagL = True
        self.last_x = None
        self.last_y = None
        self.weightN = None
        self.last_n = None

        self.c1 = self.grplot.plot(clipToView=True) # c1 is the population average
        self.c1_stim = [self.grplot.plot(clipToView=True) for _ in range(len(self.COLOR))]
        self.c2 = self.grplot_2.plot() # c2 is the individual neuron
        grplot = [self.grplot, self.grplot_2]
        for plt in grplot:
            plt.getAxis('bottom').setTickSpacing(major=50, minor=50)
        self.updateLines()
        self.activePlot = 'r'

        #videos
        self.rawplot.ui.histogram.vb.setLimits(yMin=-0.1, yMax=200) #0-255 needed, saturated here for easy viewing

    def _runProcess(self):
        '''Run ImageProcessor in separate thread
        '''
        self.comm.put([Signal.run()])
        logger.info('-------------------------   put run in comm')

    def _setup(self):
        self.comm.put([Signal.setup()])
        self.visual.setup()
    
    def updateVideo(self):
        ''' TODO: Bug on clicking ROI --> trace and report to pyqtgraph
        '''
        raw, color = self.visual.getFrames() # gets the raw frame and color of the frame
        # I think color is a masked array, where the neuron location is the only part with color?

        # raw_plot is the Raw Image
        # raw_plot2 is the processed image with the color frame over the neuron, tuned group

        if raw is not None:
            raw = raw.T  # may need to change this for accurate Bruker transformations
            # plotting the image if the size of the frame is larger than 1
            if np.unique(raw).size > 1:
                self.rawplot.setImage(raw) #, autoHistogramRange=False)
                self.rawplot.ui.histogram.vb.setLimits(yMin=80, yMax=200)

        # this colored frame goes over the raw image to plot a neuron
        if color is not None:
            color = color.T
            self.rawplot_2.setImage(color)

        # if there are photostimmed neurons, plot these
        if self.visual.stimmed_neurons is not None:
            self.updateStimmedNeurons(self.visual.stimmed_neurons, 
                                      self.visual.stimmed_xcoords, self.visual.stimmed_ycoords)


    def updateLines(self):
        ''' Helper function to plot the line traces of the activity of the selected neurons.
        '''
        penW=pyqtgraph.mkPen(width=2, color='w')
        penR=pyqtgraph.mkPen(width=2, color='r')

        C = None
        Cx = None
        try:
            # default selected neuron is 0, listed above 
            _selected_neuron = np.random.randint(0, len(self.visual.stimmed_neurons)) # for now just trying to show a random neuron that was stimulated
            self.selected_neuron = self.visual.stimmed_neurons[_selected_neuron] # neuron id in the stimmed neuron list
            logger.info('neuron trace is from cell id {}'.format(self.selected_neuron))

            (Cx, C, Cpop) = self.visual.getCurves(self.selected_neuron) # gather the tuning curves and fluorescence traces
        except TypeError:
            pass
        except Exception as e:
            logger.error('Output does not likely exist. Error: {}'.format(e))

        if (C is not None and Cx is not None):
            self.c1.setData(Cx, Cpop, pen=penW)

            # plots the color bar over each stimulus event, with color of binocular motion from above COLOR dict
            for i, plot in enumerate(self.c1_stim):
                try:
                    if len(self.visual.allStims[i]) > 0:
                        d = []
                        for s in self.visual.allStims[i]:
                            d.extend(np.arange(s,s+10).tolist())
                        display = np.clip(d, np.min(Cx), np.max(Cx))
                        try:
                            plot.setData(display, [int(np.max(Cpop))+1] * len(display),
                                    symbol='s', symbolSize=6, antialias=False,
                                    pen=None, symbolPen=self.COLOR[i], symbolBrush=self.COLOR[i])
                        except:
                            print(display)
                    if i==8 and len(self.visual.stimTimes) > 0:
                        d = []
                        for s in self.visual.stimTimes:
                            d.extend(np.arange(s,s+10).tolist())
                        display = np.clip(d, np.min(Cx), np.max(Cx))
                        try:
                            plot.setData(display, [int(np.max(Cpop))+1] * len(display),
                                    symbol='s', symbolSize=6, antialias=False,
                                    pen=None, symbolPen=self.COLOR[8], symbolBrush=self.COLOR[8])
                        except:
                            print(display)
                except KeyError:
                    pass

            self.c2.setData(Cx, C, pen=penR)
        
    def mouseClick(self, event):
        '''Clicked on processed image to select neurons
        '''
        event.accept()
        mousePoint = event.pos()
        self.selected = self.visual.selectNeurons(int(mousePoint.x()), int(mousePoint.y()))
        selectedraw = np.zeros(2)
        selectedraw[0] = int(mousePoint.x())
        selectedraw[1] = int(mousePoint.y())
        self._updateRedCirc()

        # if self.last_n is None:
        #     self.last_n = self.visual.selectedNeuron
        # elif self.last_n == self.visual.selectedNeuron:
        #     for i in range(18):
        #         self.rawplot_2.getView().removeItem(self.lines[i])
        #     self.flagW = True


    ## Anne's code for the updated red circle (only will show one neuron)
    def _updateRedCirc(self, x, y):
        ''' Circle neuron whose activity is in top (red) graph
            Default is neuron #0 from initialize
            Plotting on both raw and processed image
            #TODO: add arg instead of self.selected
        '''
        ROIpen1=pyqtgraph.mkPen(width=1, color='r')
        if self.flag:
            self.red_circ = CircleROI(pos = np.array([x, y])-5, size=10, movable=False, pen=ROIpen1)
            self.rawplot_2.getView().addItem(self.red_circ) # plotting on the processed image
            self.red_circ2 = CircleROI(pos = np.array([x, y])-5, size=10, movable=False, pen=ROIpen1)
            self.rawplot.getView().addItem(self.red_circ2) # plotting on the raw image
            self.flag = False
        else:
            self.rawplot_2.getView().removeItem(self.red_circ)
            self.rawplot.getView().removeItem(self.red_circ2)
            self.red_circ = CircleROI(pos = np.array([x, y])-5, size=10, movable=False, pen=ROIpen1)
            self.rawplot_2.getView().addItem(self.red_circ)
            self.red_circ2 = CircleROI(pos = np.array([x, y])-5, size=10, movable=False, pen=ROIpen1)
            self.rawplot.getView().addItem(self.red_circ2)


    def updateStimmedNeurons(self, x_lst = None, y_lst = None):
        ''' 
        Circle neurons that were photostimulated
        x_lst - list of x coordinates of photostimulated neurons
        y_lst - list of y coordinates of photostimulated neurons
        '''
        if x_lst is None:
            # default is a (0,0) coordinate 
            x_lst = [0]
            y_lst = [0]

        ROIpen1=pyqtgraph.mkPen(width=1, color='r') # red color 

        for x, y in zip(x_lst, y_lst):
            # to draw the circles on the Raw image only
            if self.flag: 
                self.red_circ = CircleROI(pos = np.array([x, y])-5, size=10, movable=False, pen=ROIpen1)
                self.rawplot.getView().addItem(self.red_circ)
                self.flag = False # keyword will change to false so that the circles can be removed next time
            
            # to remove the circles and then draw again on the Raw image only
            else: 
                self.rawplot.getView().removeItem(self.red_circ)
                self.red_circ = CircleROI(pos = np.array([x, y])-5, size=10, movable=False, pen=ROIpen1)
                self.rawplot.getView().addItem(self.red_circ)

    def closeEvent(self, event):
        '''Clicked x/close on window
            Add confirmation for closing without saving
        '''
        confirm = QMessageBox.question(self, 'Message', 'Stop the experiment?',
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirm == QMessageBox.Yes:
            self.comm.put(['stop'])
            print('Visual got through ', self.visual.frame_num, ' frames')
            np.savetxt('output/timing/visual_frame_time.txt', np.array(self.visual.total_times))
            np.savetxt('output/timing/gui_frame_time.txt', np.array(self.total_times))
            np.savetxt('output/timing/visual_timestamp.txt', np.array(self.visual.timestamp))
            event.accept()
        else: event.ignore()

def _call(fnc, *args, **kwargs):
    ''' Call handler for (external) events
    '''
    def _callback():
        return fnc(*args, **kwargs)
    return _callback

class CircleROI(EllipseROI):
    '''
    Makes the circle ROI in pyqtgraph
    '''
    def __init__(self, pos, size, **args):
        pyqtgraph.ROI.__init__(self, pos, size, **args)
        self.aspectLocked = True

class PolyROI(PolyLineROI):
    def __init__(self, positions, pos, **args):
        closed = True
        print('got positions ', positions)
        pyqtgraph.ROI.__init__(self, positions, closed, pos, **args)

def cmapToColormap(cmap: ListedColormap) -> ColorMap:
    """ Converts matplotlib cmap to pyqtgraph ColorMap. """

    colordata = (np.array(cmap.colors) * 255).astype(np.uint8)
    indices = np.linspace(0., 1., len(colordata))
    return ColorMap(indices, colordata)


if __name__=="__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    rasp = FrontEnd(None,None)
    rasp.show()
    app.exec_()
