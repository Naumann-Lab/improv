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
# (1) describe the stimulus set you want and (2) select from that set randomly

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
        np.save('output/optimized_neurons.npy', np.array(self.optimized_n))
        # print(self.stopping_list)
        np.save('output/stopping_list.npy', np.array(self.stopping_list))
        # print(self.peak_list)
        np.save('output/peak_list.npy', np.array(self.peak_list))
        # print(self.optim_f_list)
        np.save('output/optim_f_list.npy', np.array(self.optim_f_list))

        print('Stimulus complete, avg time per frame: ', np.mean(self.total_times))
        print('Stim got through ', self.frame_num, ' frames')
        
    def runStep(self):
        ### Get data from analysis actor
        try:
            ids = self.q_in.get(timeout=0.0001)

            X, Y, stim, _ = self.client.getList(ids)
            tmpX = np.squeeze(np.array(X)).T
            # print(tmpX.shape, '----------------------------------------------------')
            sh = len(tmpX.shape)
            if sh > 1:
                self.X = tmpX.copy()
                if tmpX.shape[1] > 8:
                    self.X = tmpX[:, -tmpX.shape[1]:]
                # print('self.X DIRECT from analysis is ', X, 'and self.X is ', self.X[:,-1])
            # print(self.X)

            try:
                b = np.zeros([len(Y),len(max(Y,key = lambda x: len(x)))])
                for i,j in enumerate(Y):
                    b[i][:len(j)] = j
                self.y0 = b.T
                # print('X, Y shapes: ', self.X.shape, self.y0.shape)
            except:
                pass
            

        except Empty as e:
            pass
        except Exception as e:
            print('Error in stimulus get: {}'.format(e))



        ### initial run ignores signals and just sends 8 basic stimuli
        if self.stop_sending:
            pass
            
        elif self.initial:
            pass


        ### once initial done, or we move on, initial GP with next neuron
        elif self.newN:
            # # ## doing random stims
            if self.random_flag:
                
                if self.prepared_frame is None:
                    self.prepared_frame = self.random_frame()
                    # self.prepared_frame.pop('load')
                if (time.time() - self.timer) >= self.total_stim_time:
                        # self.random_frame()
                    self.send_frame(self.prepared_frame)
                    self.prepared_frame = None

            else:
                # print(self.optimized_n, set(self.optimized_n))
                nonopt = np.array(list(set(np.arange(self.y0.shape[0]))-set(self.optimized_n)))
                print('nonopt is ', nonopt, ' number of neurons ', self.y0.shape[0])
                if len(nonopt) >= 1 or len(self.goback_neurons)>=1:
                    if len(nonopt) >= 1:
                        self.nID = nonopt[np.argmax(np.mean(self.y0[nonopt,:], axis=1))]
                        print('selecting most responsive neuron: ', self.nID)
                        self.optimized_n.append(self.nID)
                        self.saved_GP_est = []
                        self.saved_GP_unc = []
                    elif len(self.goback_neurons)>=1:
                        self.nID = self.goback_neurons.pop(0)
                        print('Trying again with neuron', self.nID)
                        self.optimized_n.append(self.nID)
                    
                    print(self.y0.shape, self.X.shape, self.X0.shape)
                    if self.X.shape[1] < self.y0.shape[1]:
                        self.optim.initialize_GP(self.X[:, :].T, self.y0[self.nID, -self.X.shape[1]:].T)
                    elif self.y0.shape[1] < self.maxT:
                        self.optim.initialize_GP(self.X[:, -self.y0.shape[1]:].T, self.y0[self.nID, -self.y0.shape[1]:].T)
                    else:
                        # self.optim.initialize_GP(self.X[:, -self.maxT:].T, self.y0[self.nID, -self.maxT:].T)
                        self.optim.initialize_GP(self.X[:, :].T, self.y0[self.nID, :].T)
                    # print('known average sigma, ', np.mean(self.optim.sigma))
                    # self.optim.initialize_GP(self.X0[:, :3], self.y0[self.nID, :3])
                    self.test_count = 0
                    self.newN = False
                    self.stopping = np.zeros(self.maxT)

                    curr_unc = np.diagonal(self.optim.sigma.reshape((576,576))).reshape((24,24))
                    curr_est = self.optim.f.reshape((24,24))
                    self.saved_GP_unc.append(curr_unc)
                    self.saved_GP_est.append(curr_est)

                    ids = []
                    # print('--------------- nID', self.nID)
                    # ids.append(self.client.put(self.nID, 'nID'))
                    ids.append(self.nID)
                    ids.append(self.client.put(curr_est, 'est'))
                    ids.append(self.client.put(curr_unc, 'unc'))
                    # ids.append(self.client.put(self.conf, 'conf'))
                    self.q_out.put(ids)
                
                else:
                    self.initial = True
                    print('----------------- done with this plane, moving to next')
                    self.send_move(10)

        ### update GP, suggest next stim
        else:
            
            if self.prepared_frame is None:
                X = np.zeros(2)
                # print('self.X from analysis is ', self.X[:,-1])
                # print('going back further', self.X)
                # print('GP_stimuli', self.GP_stimuli[0])
                # try:
                X[0] = self.GP_stimuli[0][int(self.X[0,-1])]
                X[1] = self.GP_stimuli[1][int(self.X[1,-1])]
                # X[2] = self.GP_stimuli[2][int(self.X[2,-1])]
                print('optim ', self.nID, ', update GP with', X, self.y0[self.nID, -1])
                self.optim.update_GP(np.squeeze(X), self.y0[self.nID,-1])

                curr_unc = np.diagonal(self.optim.sigma.reshape((576,576))).reshape((24,24))
                curr_est = self.optim.f.reshape((24,24))
                self.saved_GP_unc.append(curr_unc)
                self.saved_GP_est.append(curr_est)
                # except:
                #     pass
                ids = []
                # print('--------------- nID', self.nID)
                # ids.append(self.client.put(self.nID, 'nID'))
                ids.append(self.nID)
                ids.append(self.client.put(curr_est, 'est'))
                ids.append(self.client.put(curr_unc, 'unc'))
                # ids.append(self.client.put(self.conf, 'conf'))
                self.q_out.put(ids)

                stopCrit = self.optim.stopping()
                print('----------- stopCrit: ', stopCrit)
                self.stopping[self.test_count] = stopCrit
                self.test_count += 1

                
                if stopCrit < 3.0e-4: #6.0e-4: #8e-2: #0.37/2.05
                    peak = self.stim_star[np.argmax(self.optim.f)]
                    print('Satisfied with this neuron, moving to next. Est peak: ', peak)
                    # self.nID += 1
                    self.newN = True
                    self.stopping_list.append(self.stopping)
                    self.peak_list.append(peak)
                    self.optim_f_list.append(self.optim.f)

                    np.save('output/saved_GP_est_'+str(self.nID)+'.npy', np.array(self.saved_GP_est))
                    np.save('output/saved_GP_unc_'+str(self.nID)+'.npy', np.array(self.saved_GP_unc))

                    if len(self.optim_f_list) >= 500:
                        print('----------------- done with this plane, moving to next')
                        self.send_move(10)
                    
                elif self.test_count >= self.maxT:
                    self.goback_neurons.append(self.nID)
                    self.newN = True
                    self.stopping_list.append(self.stopping)
                    peak = self.stim_star[np.argmax(self.optim.f)]
                    self.peak_list.append(peak)
                    self.optim_f_list.append(self.optim.f)

                else:
                    ind, xt_1 = self.optim.max_acq()
                    print('suggest next stim: ', ind, xt_1, xt_1.T[...,None].shape)
                    self.prepared_frame = self.create_chosen_stim(ind)
 
            if (time.time() - self.timer) >= self.total_stim_time:
                self.send_frame(self.prepared_frame)
                self.prepared_frame = None

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
        self._socket.send_string('move', zmq.SNDMORE)
        self._socket.send_pyobj(z)
        print('sent move command')

    def create_chosen_stim(self, ind):
        xt = self.stim_star[ind]
        angle = xt[0]
        angle2 = xt[1]
        stim = self.create_frame(angle, angle2)
        return stim

    def create_frame(self, angle, angle2):
        ### Static or common stimulus params are set here
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
        if self.which_angle%8 == 0:
            random.shuffle(self.initial_angles)
        angle = self.initial_angles[self.which_angle%8] #self.stim_sets[0][self.which_angle%len(self.stim_sets[0])]
        
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


    def random_frame(self):
        ## grid choice
        snum = int(self.stim_choice[0] / 2)
        grid = np.argwhere(self.grid_choice==self.grid_ind[self.displayed_stim_num%(snum**2)])[0] #self.which_angle%24 #self.which_angle%24 #np.argwhere(self.grid_choice==self.grid_ind[self.displayed_stim_num%(36*36)])[0]
        angle = self.stimuli[0][grid[0]] #self.all_angles[grid] #self.stimuli[0][grid[0]]
        angle2 = self.stimuli[0][grid[1]]

        stim = self.create_frame(angle, angle2)
        self.timer = time.time()
        return stim


