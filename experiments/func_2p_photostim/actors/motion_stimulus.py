import time
import numpy as np
import random
import zmq
from improv.actor import Actor
from queue import Empty
from scipy.stats import norm
import random

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# goal: show visual stimuli of my choosing
# (1) describe the stimulus set you want (can either be a npy array that you load in or can set up in this actor)
# (2) select from that set in an order (write an orderstims function)
# (3) send the stimulus over zmq
# (4) wait for the whole stimulus to run before sending the next one
# (5) do a try and except to make sure that this runstep fails a lot


class VisualStimulus(Actor):

    def __init__(self, *args, ip=None, port=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ip = ip
        self.port = port
        self.frame_num = 0
        self.displayed_stim_num = 0
        self.stop_sending = False

        self.prepared_frame = None
        self.random_flag = False

        self.initial = True
        self.newN = False

        ## random sampling for initialization
        self.initial_length = 16

    def setup(self):
        context = zmq.Context()
        
        print('Starting setup')
        self._socket = context.socket(zmq.PUB)
        send_IP =  self.ip
        send_port = self.port
        self._socket.bind('tcp://' + str(send_IP)+":"+str(send_port))
        self.stimulus_topic = 'stim'
        print('Done setup VisStim')

        self.timer = time.time()
        self.total_times = []
        self.timestamp = []
        self.stimmed = []
        self.frametimes = []
        self.framesendtimes = []
        self.stimsendtimes = []
        self.tailsendtimes = []
        self.tails = []

    def stop(self):
        '''Triggered at Run
        '''
        np.save('output/stopping_list.npy', np.array(self.stopping_list))
        np.save('output/peak_list.npy', np.array(self.peak_list))
        np.save('output/optim_f_list.npy', np.array(self.optim_f_list))

        print('Stimulus complete, avg time per frame: ', np.mean(self.total_times))
        print('Stim got through ', self.frame_num, ' frames')
        
    def runStep(self):

        ### initial run ignores signals and just sends 8 basic stimuli
        if self.stop_sending:
            pass
            
        elif self.initial:
            if self.prepared_frame is None:
                self.prepared_frame = self.initial_frame()
                # logger.info(self.prepared_frame)
                # self.prepared_frame.pop('load')
            if (time.time() - self.timer) >= self.total_stim_time: # waits for the stimulus to do its thing before sending the next one
                # logger.info(self.prepared_frame)
                self.send_frame(self.prepared_frame)
                self.prepared_frame = None
        else:
            pass
            
    def send_frame(self, stim):
        if stim is not None:
            text = {'frequency':30, 'dark_value':0, 'light_value':250, 'texture_size':(1024,1024), 'texture_name':'grating_gray'}
            stimulus = {'stimulus': stim, 'texture': [text, text]}
            self._socket.send_string(self.stimulus_topic, zmq.SNDMORE)
            self._socket.send_pyobj(stimulus)
            self.timer = time.time()
            print('Number of stimuli displayed: ', self.displayed_stim_num)
            self.displayed_stim_num += 1

    def send_move(self, z):
        '''
        send the stimuli to move command over zmq
        '''
        self._socket.send_string('move', zmq.SNDMORE)
        self._socket.send_pyobj(z)
        print('sent move command')


    def create_frame(self, angle, angle2):
        '''
        How to make a motion frame for pandastim to use
        Static or common stimulus params are set here
        '''
    
        vel = -0.01
        freq = 30
        light, dark = 250, 0

        stat_t = 10
        stim_t = stat_t + 5
        self.total_stim_time = stim_t
        center_width = 12
        center_x = 0
        center_y = 0
        strip_angle = 0

        stim = {
                'stim_name': 'stim_name',
                'angle': (angle, angle2),
                'velocity': (vel, vel),
                'stationary_time': (stat_t, stat_t),
                'duration': (stim_t, stim_t),
                'frequency': (freq, freq),
                'light_value': (light, light),
                'dark_value': (dark, dark),
                'strip_width' : center_width,
                'position': (center_x, center_y),
                'strip_angle': strip_angle
                    }

        self.timer = time.time()
        return stim

    def initial_frame(self):
        '''
        the initialization step of visual motion stimuli presentation, I want to make this play an ordered stimulus set
        which_angle is the counter of stims
        '''
        if self.which_angle%8 == 0:
            random.shuffle(self.initial_angles)
        angle = self.initial_angles[self.which_angle%8] 
        
        self.which_angle += 1
        if self.which_angle >= self.initial_length: 
            self.initial = True #False
            self.stop_sending = True
            # self.newN = True
            self.which_angle = 0
            logger.info('Done with initial frames, starting random set')
        
        stim = self.create_frame(angle, angle)
        self.timer = time.time()
        return stim



