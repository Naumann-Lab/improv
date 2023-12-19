import os
import time
import h5py
import numpy as np
import zmq
import json
import pathlib
from pathlib import Path
from improv.actor import Actor, RunManager
from ast import literal_eval as make_tuple

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ZMQAcquirer(Actor):

    def __init__(self, *args, ip=None, ports=None, output=None, red_chan_image=None, init_filename=None, init_frame=120, **kwargs):
        super().__init__(*args, **kwargs)
        print("init")
        self.ip = ip
        self.ports = ports
        self.frame_num = 0
        self.stim_count = 0
        self.initial_frame_num = init_frame     # Number of frames for initialization
        ## FIXME
        self.init_filename = init_filename #'output/initialization.h5'
        self.red_chan_image = red_chan_image
        
        self.output_folder = str(output)
        pathlib.Path(output).mkdir(exist_ok=True) 
        pathlib.Path(output+'timing/').mkdir(exist_ok=True)

    def setup(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        for port in self.ports:
            self.socket.connect("tcp://"+str(self.ip)+":"+str(port))
            print('Connected to '+str(self.ip)+':'+str(port))
        self.socket.setsockopt(zmq.SUBSCRIBE, b'')

        self.saveArray = []
        self.saveArrayRedChan = []
        self.save_ind = 0
        self.fullStimmsg = []
        self.total_times = []
        self.timestamp = []
        self.stimmed = []
        self.frametimes = []
        self.framesendtimes = []
        self.stimsendtimes = []
        self.tailsendtimes = []
        self.tails = []

        self.tailF = False
        self.stimF = False
        self.frameF = False
        self.align_flag = True

        if not os.path.exists(self.init_filename):

            ## Save initial set of frames to output/initialization.h5
            self.kill_flag = False
            while self.frame_num < self.initial_frame_num:
                self.runStep()

            self.imgs = np.array(self.saveArray)
            f = h5py.File(self.init_filename, 'w', libver='earliest')
            f.create_dataset("default", data=self.imgs)
            f.close()

        if not os.path.exists(self.red_chan_image):
            if len(self.saveArrayRedChan) > 1:
                mean_red = np.mean(np.array(self.saveArrayRedChan), axis=0)
                np.save(self.red_chan_image, mean_red)

        self.frame_num = 0

        self.kill_flag = True

        # ## reconnect socket
        # self.socket.close()
        # self.socket = context.socket(zmq.SUB)
        # for port in self.ports:
        #     self.socket.connect("tcp://"+str(self.ip)+":"+str(port))
        #     print('RE-Connected to '+str(self.ip)+':'+str(port))
        # self.socket.setsockopt(zmq.SUBSCRIBE, b'')


    def stop(self):
        self.imgs = np.array(self.saveArray)
        f = h5py.File('output/sample_stream_end.h5', 'w', libver='earliest')
        f.create_dataset("default", data=self.imgs)
        f.close()

        np.savetxt('output/stimmed.txt', np.array(self.stimmed))
        np.save('output/tails.npy', np.array(self.tails))
        np.savetxt('output/timing/frametimes.txt', np.array(self.frametimes))
        np.savetxt('output/timing/framesendtimes.txt', np.array(self.framesendtimes), fmt="%s")
        np.savetxt('output/timing/stimsendtimes.txt', np.array(self.stimsendtimes), fmt="%s")
        np.savetxt('output/timing/tailsendtimes.txt', np.array(self.tailsendtimes), fmt="%s")
        np.savetxt('output/timing/acquire_frame_time.txt', self.total_times)
        np.savetxt('output/timing/acquire_timestamp.txt', self.timestamp)
        np.save('output/fullstim.npy', self.fullStimmsg)

        print('Acquisition complete, avg time per frame: ', np.mean(self.total_times))
        print('Acquire got through ', self.frame_num, ' frames')

    def runStep(self):

        if self.kill_flag:
            self.socket.close()
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.SUB)
            for port in self.ports:
                self.socket.connect("tcp://"+str(self.ip)+":"+str(port))
                print('RE-Connected to '+str(self.ip)+':'+str(port))
            self.socket.setsockopt(zmq.SUBSCRIBE, b'')

            timer = time.time()
            while True:
                msg = self.socket.recv_multipart()
                if time.time()-timer > 0.02:
                    self.kill_flag = False
                    logger.error('Resuming run') 
                    break
                logger.info(str(time.time()-timer))
                timer = time.time()


            # cnt = 0
            # while True:
            #     try:
            #         msg = self.socket.recv_multipart()
            #         logger.info('read and threw away')
            #         cnt += 1
            #     except zmq.Again:
            #         self.kill_flag = False
            #         logger.error('Resuming run after {} iterations'.format(cnt)) 
            #         break

            self.kill_flag = False

        #     # timer = time.time()
        #     while True:
        #         try:
        #             msg = self.socket.recv_multipart()
        #         except zmq.Again:
        #             self.kill_flag = False
        #             logger.error('Resuming run') 
        #             break
        #     # if time.time()-timer > 0.05:
        #         # self.kill_flag = False
        #         # logger.error('Resuming run') 

        try:
            self.get_message()
        except zmq.Again:
            # No messages available
            pass 
        except Exception as e:
            print('error: {}'.format(e))

    def get_message(self, timeout=0.001):
        msg = self.socket.recv_multipart() #(flags=zmq.NOBLOCK)

        try:
            msg_dict = self._msg_unpacker(msg)
            tag = msg_dict['type'] 
        except:
            logger.error('Weird format message {}'.format(msg))

        if'stim' in tag: 
            if not self.stimF:
                logger.info('Receiving stimulus information')
                self.stimF = True
            self.fullStimmsg.append(msg)
            self._collect_stimulus(self, msg_dict)

        elif 'frame' in tag: 
            t0 = time.time()
            self._collect_frame(msg_dict)
            self.frame_num += 1
            self.total_times.append(time.time() - t0)

        elif str(tag) in 'tail':
            if not self.tailF:
                logger.info('Receiving tail information')
                self.tailF = True
            self._collect_tail(msg_dict)

    def _collect_frame(self, msg_dict):
        array = np.array(json.loads(msg_dict['data']))
        if not self.frameF:
            logger.info('Receiving frame information')
            self.frameF = True
            logger.info('Image frame(s) size is {}'.format(array.shape))
            if array.shape[0] == 2:
                logger.info('Acquiring also in the red channel')
                # obj_id = self.client.put(array[1], 'acq_red' + str(self.frame_num))
                # self.links['red_channel'].put({self.frame_num: obj_id})
        self.saveArray.append(array[0])
        if array.shape[0] == 2:
            self.saveArrayRedChan.append(array[1])
        
        if not self.align_flag:
            array = None
        obj_id = self.client.put(array[0], 'acq_raw' + str(self.frame_num))
        self.q_out.put([{str(self.frame_num): obj_id}])

        sendtime =  msg_dict['timestamp'] 

        self.frametimes.append([self.frame_num, time.time()])
        self.framesendtimes.append([sendtime])

        if len(self.saveArray) >= 1000:
            self.imgs = np.array(self.saveArray)
            f = h5py.File(self.output_folder+'/sample_stream'+str(self.save_ind)+'.h5', 'w', libver='earliest')
            f.create_dataset("default", data=self.imgs)
            f.close()
            self.save_ind += 1
            del self.saveArray
            self.saveArray = []

    def _collect_stimulus(self, msg_dict):
        sendtime = msg_dict['time']

        category = str(msg_dict['raw_msg']) #'motionOn' 
        if 'alignment' in category:
            ## Currently not using
            s = msg_dict[5]
            status = str(s.decode('utf8').encode('ascii', errors='ignore'))
            if 'start' in status:
                self.align_flag = False
                logger.info('Starting alignment...')
            elif 'completed' in status:
                self.align_flag = True
                print(msg_dict)
                logger.info('Alignment done, continuing')
        elif 'move' in category:
            pass 
            # print(msg)  
        elif 'motionOn' in category:
            angle2 = None
            angle, angle2 = make_tuple(msg_dict['angle'])
            if angle>=360:
                angle-=360
            stim = self._realign_angle(angle)
            self.stim_count += 1

            logger.info('Stimulus: {}, angle: {},{}, frame {}'.format(stim, angle, angle2, self.frame_num))
            logger.info('Number of stimuli displayed: {}'.format(self.stim_count))
            
            self.links['stim_queue'].put({self.frame_num:[stim, float(angle), float(angle2)]})

            self.stimmed.append([self.frame_num, stim, angle, angle2, time.time()])
            self.stimsendtimes.append([sendtime])

    def _collect_tail(self, msg_dict):
        sendtime = msg_dict['timestamp']
        tails = np.array(msg_dict['tail_points']) 
        self.tails.append(tails) 
        self.tailsendtimes.append([sendtime])

    def _msg_unpacker(self, msg):
        keys = msg[::2]
        vals = msg[1::2]

        msg_dict = {}
        for k, v in zip(keys, vals):
            msg_dict[k.decode()] = v.decode()
        return msg_dict

    def _realign_angle(self, angle):
        if 23 > angle >=0:
            stim = 9
        elif 360 > angle >= 338:
            stim = 9
        elif 113 > angle >= 68:
            stim = 3
        elif 203 > angle >= 158:
            stim = 13
        elif 293 > angle >= 248:
            stim = 4
        elif 68 > angle >= 23:
            stim = 10
        elif 158 > angle >= 113:
            stim = 12
        elif 248 > angle >= 203:
            stim = 14
        elif 338 > angle >= 293:
            stim = 16
        else:
            logger.error('Stimulus angle unrecognized')
            stim = 0
        return stim
