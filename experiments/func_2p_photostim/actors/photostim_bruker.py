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

import sys
sys.path.append('/home/user/Code/bruker2P_control') 
from markpoints import ZMQ_Photostim_Client

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PhotoStimulus(Actor):

    def __init__(self, *args, ip=None, port=None, seed=1234, stimuli=None, selected_group = None, **kwargs): 
        super().__init__(*args, **kwargs)
        self.ip = ip
        self.port = port
        self.frame_num = 0

        self.selected_group = selected_group
        if self.selected_group is None:
            self.selected_group = 'forward' # default tuning is Forward

        self.seed = 1337 # for consistent testing
        np.random.seed(self.seed) 

        self.photostim_output = {} # will be saved later as the photostim output file

        self.photostim_client = ZMQ_Photostim_Client.Photostim_Client()

    def setup(self):
        context = zmq.Context()
        
        self.socket = context.socket(zmq.PUB)
        send_IP =  self.ip
        send_port = self.port
        self.socket.bind('tcp://' + str(send_IP)+":"+str(send_port))
        self.stimulus_topic = 'stim' # change this to be a different keyword?

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
        np.save('output/photostims.npy', np.array(self.photostim_output))
        
    def runStep(self):       

        ### Get data from analysis actor and gui 'actor' aka the visual_viz_stim
        params_dict = None
        stim_coords = None
        try:
            # visual actor (via the gui) sends in a photostim params dictionary
            q_params_dict = self.q_params_in.get(timeout=0.0001) 
            params_dict = self.client.getList(q_params_dict)[0] # dictionary is in a list, so need to index into it

            analyzed_cells = self.q_in.get(timeout=0.0001) # analysis actor sends in full info about all neuron functional identity
            
            # what I want to get from analysis actor:
            (C, tune, coords, planes, color) = self.client.getList(analyzed_cells[:-1]) # index -1 is the frame number
            # coords = array of coordinates
            # tune = array of tuning categories for each neuron
            # C is calcium activity over time for each neuron
            # color = color of neuron
            # planes = plane that cell is on
        
            # collecting the coordinates to be stimulated based on specified functional identity
            tune_arr = np.array(tune)
            neur_ids = np.where(tune_arr == self.selected_group) # the cell ids that correspond to the selected functional identity
            stim_coords = coords[neur_ids] # indexing into the coordinate array with the inds of the tuned neurons
            # coordinates may need to be swapped??
            params_dict['points'] = stim_coords # check if this is an array or list...
            
            # needs to be the neuron index from the analysis to grab the neuron for plotting
            params_dict['neur_id'] = neur_ids 

            # specify galvo version
            if params_dict['procedure'] == 'galvo':
                if len(neur_ids) > 1: 
                    params_dict['procedure'] = 'slm-2d'

            if (params_dict is not None) & (stim_coords is not None): # only stimulate if you are sent a parameter dictionary and have coords to stimulate
                logger.info('{}'.format('photostim is starting'))
                # will be updated by Owen
                self.photostim_output.append(np.array(['start_photostim', time.time()])) # mark when the photostim starts in time stamp
                
                photostim_info = self.photostim_client.send_request(params_dict) # sends params dict to PL, send back information about the photostim event
                logger.info('{}'.format(photostim_info)) # edit if needed 

                self.photostim_output.append(np.array(['finish_photostim', self.frame_num, self.selected_group, self.params_dict, time.time()]))
                logger.info('{}'.format('photostim success'))

                # output is list of neuron ids, x coords, y coords, functional group, and frame number
                self.q_out.put([neur_ids, stim_coords[:, 0], stim_coords[:, 1], self.selected_group, self.frame_num]) 

                if len(stim_coords) == 0: # debugging - if the length of stim coord list is 0, then no stimulation
                    logger.info('{}'.format('no photostim today'))
                    self.photostim_output.append(np.array(['no_photostim', time.time()]))

        except Empty as e:
            pass
        except Exception as e:
            logger.error('Error in stimulus get: {}'.format(e))
        
