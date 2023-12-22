import time
import os
import sys
import h5py
import struct
import numpy as np
import ipaddress
import zmq
import json
from pathlib import Path
from improv.actor import Actor, RunManager
from queue import Empty
from scipy.stats import norm
import random

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PhotoStimulus(Actor):

    def __init__(self, *args, ip=None, port=None, seed=1234, stimuli=None, selected_tune = None, **kwargs): 
        super().__init__(*args, **kwargs)
        self.ip = ip
        self.port = port
        self.frame_num = 0
        self.displayed_stim_num = 0
        self.selected_neuron = None
        self.rep_count = 3
        self.img_size = 400

        self.selected_tune = selected_tune
        if self.selected_tune is None:
            self.selected_tune = 'forward' # default tuning is Forward

        self.seed = 1337 # what is seed for?
        np.random.seed(self.seed)

        self.photo_frames = [] # will be saved later as the photostim output file

    def setup(self):
        context = zmq.Context()
        
        self.socket = context.socket(zmq.PUB)
        send_IP =  self.ip
        send_port = self.port
        self.socket.bind('tcp://' + str(send_IP)+":"+str(send_port))
        self.stimulus_topic = 'stim'

        self.stimmed_neurons = []

        self.timer = time.time()
        self.total_times = []
        self.timestamp = []
        self.stimmed = []
        self.frametimes = []
        self.framesendtimes = []
        self.stimsendtimes = []
        self.whole_timer = time.time()

    def stop(self):
        print('Stimulus complete, avg time per frame: ', np.mean(self.total_times))
        print('Stim got through ', self.frame_num, ' frames')
        np.save('output/photostims.npy', np.array(self.photo_frames))
        
    def runStep(self):       

        ### Get data from analysis actor and gui 'actor' aka the visual viz stim
        params_dict = None
        stim_coords = None
        try:
            # visual actor (via the gui) sends in a photostim params dictionary
            _params_dict = self.q_params_in.get(timeout=0.0001) 
            params_dict = self.client.getList(_params_dict)[0] # dictionary is in a list, so need to index into it

            ids = self.q_in.get(timeout=0.0001) # analysis actor sends in full info about all neuron functional identity
            # what I want to get:
            (C, tune, coords, planes, color) = self.client.getList(ids[:-1]) # index -1 is the frame number
            # coords = array of coordinates, tune = array of tuning categories for each neuron, C is calcium activity over time for each neuron, color = color of neuron, planes = plane that cell is on
        
            # collecting the coordinates to be stimulated based on specified functional identity
            tune_arr = np.array(tune)
            neur_ids = np.where(tune_arr == self.selected_tune)[0]
            stim_coords = coords[neur_ids] # coordinates may be swapped??
            params_dict['points'] = stim_coords
            
            # needs to be the neuron index from the analysis to grab the neuron for plotting
            params_dict['neur_id'] = neur_ids 

            # specify galvo version
            if params_dict['procedure'] == 'galvo':
                if len(neur_ids) > 1: 
                    params_dict['procedure'] = 'galvo_2D'


            if (params_dict is not None) & (stim_coords is not None): # only stimulate if you are sent a parameter dictionary and have coords to stimulate
                logger.info('{}'.format('photostim is starting'))
                self.photo_frames.append(np.array(['start_photostim', time.time()])) # mark when the photostim starts in time stamp
                self.photostim_via_PL(params_dict)
                self.photo_frames.append(np.array(['finish_photostim', self.frame_num, self.selected_tune, params_dict, time.time()]))
                logger.info('{}'.format('photostim success'))
                self.q_out.put([neur_ids, stim_coords[:, 0], stim_coords[:, 1], self.frame_num])
                if len(stim_coords) == 0:
                    logger.info('{}'.format('no photostim today'))
                    self.photo_frames.append(np.array(['no_photostim', time.time()]))
        except Empty as e:
            pass
        except Exception as e:
            logger.error('Error in stimulus get: {}'.format(e))


    ### TODO: put in photostimulation script here to run on the Bruker
            
    def photostim_via_PL(self):
        ####
        sys.path.append('/home/user/Code/bruker2P_control') 
        from markpoints import ZMQ_Photostim_Client
        # does this function handle both galvos and slm?

## old neuron selection criteria
# logger.info('got to pici stim neuron')
        # # ll = np.array(tc_list)
        # # ll[ll==np.inf] = 0
        # # ll[ll<0] = 0
        # # left = np.array([-1,-1,1,0])
        # # candidate = np.argmax(ll.dot(left))
    
        # # if self.counter_stim >= self.rep_count or self.displayed_stim_num < 1:
        # #     index = candidate #random.choice(neurons) # candidate #.shape[0])
        # #     # print('tuning curve is ', tune[candidate], 'candidate ', candidate, ' color ', ll[candidate])
        # #     self.selected_neuron = index
        # #     self.counter_stim = 0
        # # else:
        # #     index = self.selected_neuron
        # # # print('Selecting neuron # '+str(index)+' for stimulation')
        # # # print('com is ', com[index])
        # # x, y = com[index][0], com[index][1]
        # r1 = [1, 3]
        # r2 = [1, 3]

        ################# 
        # ll = np.array(tc_list)
        # ll[ll==np.inf] = 0
        # ll[ll<0] = 0
        # left = np.array([-1,-1,1,0])
        # right = np.array([1,-1,-1,0])
        # dotted = ll.dot(right) #left)
        # possibles = np.squeeze(np.argwhere(np.abs(dotted - dotted.max()) < 75)) #50)
        # # candidate = np.argmax(ll.dot(left))
        # if len(self.photostimmed_neurons) > 0:
        #     # print(possibles)
        #     # print(self.photostimmed_neurons)
        #     # print(np.append(possibles, self.photostimmed_neurons))
        #     pp = np.unique(np.append(possibles, self.photostimmed_neurons))
        #     # print(pp)
        # else: pp = possibles