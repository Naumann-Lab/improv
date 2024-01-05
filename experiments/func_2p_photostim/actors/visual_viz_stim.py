import time
import numpy as np
from scipy.spatial.distance import cdist
from queue import Empty
from collections import deque
from PyQt5 import QtWidgets

from improv.actor import Actor, Signal
from improv.store import ObjectNotFoundError

# this is Anne's gui imported here 
# from .GUI import FrontEnd 

# need this import for my GUI to work
from .GUI_paramset import FrontEnd 

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DisplayVisual(Actor):
    ''' Class used to run a GUI + Visual as a single Actor 
    '''
    def run(self):
        logger.info('Loading FrontEnd')
        self.app = QtWidgets.QApplication([])
        self.rasp = FrontEnd(self.visual, self.q_comm)
        self.rasp.show()
        logger.info('GUI ready')
        self.q_comm.put([Signal.ready()])
        self.visual.q_comm.put([Signal.ready()])
        self.app.exec_()
        logger.info('Done running GUI')

    def setup(self, visual=None):
        logger.info('Running setup for '+self.name)
        self.visual = visual
        self.visual.setup()

class CaimanVisualStim(Actor):
    ''' Class for displaying data from caiman processor
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        self.com1 = np.zeros(2)
        self.selectedNeuron = 0 # default is the first neuron
        self.selected_group = 'forward' # default tuning is forward, can change this
        self.frame_num = 0

        self.red_chan = None
        self.stimTimes = []

    def setup(self):
        # these variables are collected from other actors
        self.Cx = None # x axis time overall
        self.C = None # fluorescence traces
        self.tune = None 
        self.raw = None
        self.color = None
        self.coords = None

        self.selected_neuron = 0 # default selected neuron is cell 0
        self.draw = True # keyword to have updated visualizations

        self.total_times = []
        self.timestamp = []

        self.window=150

        self.red_chan = None # don't have red channel images yet

    def run(self):
        pass #NOTE: Special case here, tied to GUI

    def getData(self):
        t = time.time()
        ids = None
        
        try: # getting raw image data to display
            id = self.links['raw_frame_queue'].get(timeout=0.0001)
            self.raw_frame_number = list(id[0].keys())[0]
            self.raw = self.client.getID(id[0][self.raw_frame_number])
        except Empty as e:
            pass
        except Exception as e:
            logger.error('Visual: Exception in get data: {}'.format(e))
        
        try: # getting information from analysis actor on functional identity and coordinates of cells
            ids = self.q_in.get(timeout=0.0001)
            if ids is not None and ids[0]==1:
                print('visual: missing frame')
                self.frame_num += 1
                self.total_times.append([time.time(), time.time()-t])
                raise Empty
            self.frame_num = ids[-1]
            if self.draw:
                (self.Cx, self.C, self.Cpop, self.tune, self.color, self.coords, self.allStims, self.tc_list) = self.client.getList(ids[:-1])
                self.total_times.append([time.time(), time.time()-t])
            self.timestamp.append([time.time(), self.frame_num])
        except Empty as e:
            pass
        except ObjectNotFoundError as e:
            logger.error('Object not found, continuing anyway...')
        except Exception as e:
            logger.error('Visual: Exception in get data: {}'.format(e))
        
        try: # getting the photostimulated neuron's information to be displayed on gui

            # this is what is received: ([neur_ids, stim_coords[:, 0], stim_coords[:, 1], self.selected_group, self.frame_num]) 
            photostim_info = self.links['optim_in'].get(timeout=0.0001) 
            self.stimmed_neurons = photostim_info[0] # neuron ids
            self.stimmed_xcoords = photostim_info[1] # x coords
            self.stimmed_ycoords = photostim_info[2] # y coords

            self.selected_group = photostim_info[3] # functional identity

            self.stimTimes.append(int(photostim_info[4])) # frame number of photostimulation
        except Empty as e:
            pass
        except Exception as e:
            logger.error('Visual: Exception in get stim for visual: {}'.format(e))

    def getCurves(self):
        ''' Return the fluorescence traces and calculated tuning curves
            for the selected neuron as well as the population average
            Cx is the time (overall or window) as x axis
            C is indexed for selected neuron and Cpop is the population avg
            tune is a similar list to C
        '''
        # get tuning curve? need to look at analysis actor to understand what 'tune' is
        if self.tune is not None:
            self.tuned = [self.selected_group, self.tune[1]]
        else:
            self.tuned = None

        # fluoresence traces of the selected neuron id
        if self.frame_num > self.window:
            self.C = self.C[:, -len(self.Cx):]
            self.Cpop = self.Cpop[-len(self.Cx):]
        
        return self.Cx, self.C[self.selected_neuron,:], self.Cpop

    def getFrames(self):
        ''' Return the raw and colored frames for display
        '''
        return self.raw, self.color


    