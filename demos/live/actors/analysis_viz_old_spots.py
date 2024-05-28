from improv.actor import Actor
from improv.store import ObjectNotFoundError
from queue import Empty
import numpy as np
import time
import cv2

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class VizStimAnalysis(Actor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args)

    def setup(self, param_file=None):
        '''
        '''
        np.seterr(divide='ignore')

        # TODO: same as behaviorAcquisition, need number of stimuli here. Make adaptive later
        self.num_stim = 12 
        self.frame = 0
        # self.curr_stim = 0 #start with zeroth stim unless signaled otherwise
        self.stim = {}
        self.stimStart = -1
        self.currentStim = None
        self.ests = np.zeros((1, self.num_stim, 2)) #number of neurons, number of stim, on and baseline
        self.counter = np.ones((self.num_stim,2))
        self.window = 150 #TODO: make user input, choose scrolling window for Visual
        self.C = None
        self.S = None
        self.Call = None
        self.Cx = None
        self.Cpop = None
        self.coords = None
        self.color = None
        self.runMean = None
        self.runMeanOn = None
        self.runMeanOff = None
        self.lastOnOff = None
        self.recentStim = [0]*self.window
        self.currStimID = np.zeros((8, 1000000)) #FIXME
        self.currStim = -10
        self.allStims = {}
        self.estsAvg = None

        self.stimX = []
        self.stimY = []
        self.testNum = 0
        self.nID = 0
        self.stimText = None

        self.total_times = []
        self.puttime = []
        self.colortime = []
        self.stimtime = []
        self.timestamp = []
        self.LL = []
        self.fit_times = []


    def stop(self):
        
        print('Analysis broke, avg time per frame: ', np.mean(self.total_times, axis=0))
        print('Analysis broke, avg time per put analysis: ', np.mean(self.puttime))
        print('Analysis broke, avg time per put analysis: ', np.mean(self.fit_times))
        print('Analysis broke, avg time per color frame: ', np.mean(self.colortime))
        print('Analysis broke, avg time per stim avg: ', np.mean(self.stimtime))
        print('Analysis got through ', self.frame, ' frames')

        np.savetxt('output/timing/analysis_frame_time.txt', np.array(self.total_times))
        np.savetxt('output/timing/analysis_timestamp.txt', np.array(self.timestamp))
        np.savetxt('output/analysis_estsAvg.txt', np.array(self.estsAvg))
        np.savetxt('output/analysis_proc_C.txt', np.array(self.C))
        
        stim = []
        for i in self.allStims.keys():
            stim.append(self.allStims[i])
        print('Stims ------------------------------')
        print(self.allStims)
        np.save('output/used_stims.npy', np.array(stim))


    def runStep(self):
        ''' Take numpy estimates and frame_number
            Create X and Y for plotting
        '''
        t = time.time()
        ids = None
        
        try:
            ids = self.q_in.get(timeout=0.0001)
            if ids is not None and ids[0]==1:
                print('analysis: missing frame')
                self.total_times.append(time.time()-t)
                self.q_out.put([1])
                raise Empty

            self.frame = ids[-1]
            (self.coordDict, self.image, self.C) = self.client.getList(ids[:-1])
            self.C = np.where(np.isnan(self.C), 0, self.C)

            if self.coordDict is not None:
                self.coords = [o['coordinates'] for o in self.coordDict]
            
            # Compute tuning curves based on input stimulus
            # Just do overall average activity for now
            try: 
                ## stim format: stim, stimonOff, angle, vel, freq, contrast
                sig = self.links['input_stim_queue'].get(timeout=0.0001)
                self.updateStim_start(sig)
                self.stimText = list(sig.values())
            except Empty as e:
                pass #no change in input stimulus

            self.stimAvg_start()
            
            self.globalAvg = np.mean(self.estsAvg[:,:8], axis=0)
            self.tune = [self.estsAvg[:,:8], self.globalAvg]

            # Compute coloring of neurons for processed frame
            # Also rotate and stack as needed for plotting
            self.color, self.tc_list = self.plotColorFrame()

            if self.frame >= self.window:
                window = self.window
                self.Cx = np.arange(self.frame-window,self.frame)
            else:
                window = self.frame
                self.Cx = np.arange(0,self.frame)

            if self.C.shape[1]>0:
                self.Cpop = np.nanmean(self.C, axis=0)
                if np.isnan(self.Cpop).any():
                    logger.error('Nan in Cpop')
                self.Call = self.C 

            self.putAnalysis()
            self.timestamp.append([time.time(), self.frame])
            self.total_times.append(time.time()-t)
        except ObjectNotFoundError:
            logger.error('Estimates unavailable from store, droppping')
        except Empty as e:
            pass
        except Exception as e:
            logger.exception('Error in analysis: {}'.format(e))
    

    def updateStim_start(self, stim):
        ''' Rearrange the info about stimulus into
            cardinal directions and frame <--> stim correspondence.

            self.stimStart is the frame where the stimulus started.
        '''
        # get frame number and stimID
        frame = list(stim.keys())[0]
        whichStim = stim[frame][0]
        # convert stimID into 8 cardinal directions
        stimID = self.IDstim(int(whichStim))
        logger.info('stimID {}'.format(stimID))

        # assuming we have one of those 8 stimuli
        if stimID != -10:
            if stimID not in self.allStims.keys():
                self.allStims.update({stimID:[]})
            # determine if this is a new stimulus trial
            # if abs(stim[frame][1])>1 :
            curStim = 1 #on
            self.allStims[stimID].append(frame)
            # else:
            #     curStim = 0 #off
            # paradigm for these trials is for each stim: [off, on, off]
            # if self.lastOnOff is None:
            #     self.lastOnOff = curStim
            # elif curStim == 1: #self.lastOnOff == 0 and curStim == 1: #was off, now on
                # select this frame as the starting point of the new trial
                # and stimulus has started to be shown
                # All other computations will reference this point
            self.stimStart = frame 
            self.currentStim = stimID
            if stimID<8:
                self.currStimID[stimID, frame] = 1
            # NOTE: this overwrites historical info on past trials
            logger.info('Stim {} started at {}'.format(stimID,frame))
            # else:
            # #     self.currStimID[:, frame] = np.zeros(8)
            # logger.info('Frame: {}, {}'.format(frame,self.lastOnOff))
            # logger.info('Current data frame is {}'.format(self.frame))

    def putAnalysis(self):
        ''' Throw things to DS and put IDs in queue for Visual
        '''
        t = time.time()
        ids = []
        ids.append(self.client.put(self.Cx, 'Cx'+str(self.frame)))
        ids.append(self.client.put(self.Call, 'Call'+str(self.frame)))
        ids.append(self.client.put(self.Cpop, 'Cpop'+str(self.frame)))
        ids.append(self.client.put(self.tune, 'tune'+str(self.frame)))
        ids.append(self.client.put(self.color, 'color'+str(self.frame)))
        ids.append(self.client.put(self.coordDict, 'analys_coords'+str(self.frame)))
        ids.append(self.client.put(self.allStims, 'stim'+str(self.frame)))
        ids.append(self.client.put(self.tc_list, 'tc_list'))
        ids.append(self.frame)

        self.q_out.put(ids)
        self.puttime.append(time.time()-t)

    def putStimulus(self):
        ids = []
        ids.append(self.client.put(self.stimX, 'stimX'+str(self.frame)))
        ids.append(self.client.put(self.stimY, 'stimY'+str(self.frame)))
        ids.append(self.client.put(self.testNum, 'stim_testNum'+str(self.frame)))
        ids.append(self.client.put(self.nID, 'stim_nID'+str(self.frame)))
        self.links['stim_out'].put(ids)

    def stimAvg_start(self):
        t = time.time()

        ests = self.C
        
        if self.ests.shape[0]<ests.shape[0]:
            diff = ests.shape[0] - self.ests.shape[0]
            # added more neurons, grow the array
            self.ests = np.pad(self.ests, ((0,diff),(0,0),(0,0)), mode='constant')
            # for key in self.ys.keys():
            #     self.ys[key] = np.pad(self.ys[key], ((0,diff),(0,0),(0,0)), mode='constant')
           
        before_amount = 5
        after_amount = 20
        if self.currentStim is not None:
            if self.stimStart == self.frame:
                # account for the baseline prior to stimulus onset
                mean_val = np.mean(ests[:,self.frame-before_amount:self.frame],1)
                self.ests[:,self.currentStim,1] = (self.counter[self.currentStim,1]*self.ests[:,self.currentStim,1] + mean_val)/(self.counter[self.currentStim,1]+1)
                self.counter[self.currentStim, 1] += before_amount

                # for key in self.ys.keys():
                #     ind = self.xs[key]
                #     try:
                #         self.ys[key][:,ind,1] = (self.counters[key][ind,1]*self.ys[key][:,ind,1] + mean_val[:,None])/(self.counters[key][ind,1]+1)
                #     except:
                #         print(key)
                #         print(self.ys[key][:,ind,1].shape)
                #         print(self.counters[key][ind,1].shape)
                #         print(mean_val.shape)
                #         print(((self.counters[key][ind,1]*self.ys[key][:,ind,1] + mean_val[:,None])/(self.counters[key][ind,1]+1)).shape)

                    # self.counters[key][ind, 1] += before_amount
                    

            elif self.frame in range(self.stimStart+1, self.stimStart+2):
                val = ests[:,self.frame-1]
                self.ests[:,self.currentStim,1] = (self.counter[self.currentStim,1]*self.ests[:,self.currentStim,1] + val)/(self.counter[self.currentStim,1]+1)
                self.counter[self.currentStim, 1] += 1

                # for key in self.ys.keys():
                #     ind = self.xs[key]
                #     try:
                #         self.ys[key][:,ind,1] = (self.counters[key][ind,1]*self.ys[key][:,ind,1] + val[:,None])/(self.counters[key][ind,1]+1)
                #     except:    
                #         print(key)
                #         print(self.ys[key][:,ind,1].shape)
                #         print(self.counters[key][ind,1].shape)
                #     self.counters[key][ind, 1] += 1

            elif self.frame in range(self.stimStart+2, self.stimStart+after_amount):
                val = ests[:,self.frame-1]
                self.ests[:,self.currentStim,0] = (self.counter[self.currentStim,0]*self.ests[:,self.currentStim,0] + val)/(self.counter[self.currentStim,0]+1)
                self.counter[self.currentStim, 0] += 1

                # for key in self.ys.keys():
                #     ind = self.xs[key]
                #     self.ys[key][:,ind,0] = (self.counters[key][ind,0]*self.ys[key][:,ind,0] + val[:,None])/(self.counters[key][ind,0]+1)
                #     self.counters[key][ind, 0] += 1

            if self.frame == self.stimStart + after_amount:
                print('appending to X: ', list(self.xs.values()))
                self.stimX.append(list(self.xs.values()))
                self.stimY.append(np.mean(ests[:,self.frame-after_amount:self.frame],1))
                self.testNum += 1
                sc = self.stim_count[int(self.xs['angle']), int(self.xs['vel'])]
                numN = self.ests.shape[0]
                self.all_y[:numN, int(self.xs['angle']), int(self.xs['vel'])] = ((sc-1)*self.all_y[:numN, int(self.xs['angle']), int(self.xs['vel'])] + self.stimY[-1])/sc


        self.estsAvg = np.squeeze(self.ests[:,:,0] - self.ests[:,:,1])        
        self.estsAvg = np.where(np.isnan(self.estsAvg), 0, self.estsAvg)
        self.estsAvg[self.estsAvg == np.inf] = 0
        self.estsAvg[self.estsAvg<0] = 0

        self.stimtime.append(time.time()-t)

    def plotColorFrame(self):
        ''' Computes colored nicer background+components frame
        '''
        t = time.time()
        image = self.image.copy()
        color = np.stack([image, image, image, image], axis=-1).astype(np.uint8).copy()
        color[...,3] = 255
        tc_list = []
            #TODO: don't stack image each time?
        if self.coords is not None:
            for i,c in enumerate(self.coords):
                pixels = c[~np.isnan(c).any(axis=1)].astype(int)
                #TODO: Compute all colors simultaneously! then index in...
                try:
                    tc = self._tuningColor(i, color[pixels[:,1], pixels[:,0]])
                    tc_list.append(tc)
                    cv2.fillConvexPoly(color, pixels, tc)
                except Exception as e:
                    # logger.error('Error in fill poly: {}'.format(e))
                    pass
                
        self.colortime.append(time.time()-t)
        return color, tc_list

    def _tuningColor(self, ind, inten):
        ''' ind identifies the neuron by number
        '''
        ests = self.estsAvg
        #ests = self.tune_k[0] 
        if ests[ind] is not None: 
            try:
                return self.manual_Color_Sum(ests[ind])                
            except ValueError:
                return (255,255,255,0)
            except Exception:
                print('inten is ', inten)
                print('ests[i] is ', ests[ind])
        else:
            return (255,255,255,50)

    def manual_Color_Sum(self, x):
        ''' x should be length 12 array for coloring
            or, for k coloring, length 8
            Using specific coloring scheme from Naumann lab
        '''
        if x.shape[0] == 8:
            mat_weight = np.array([
            [1, 0.25, 0],
            [0.75, 1, 0],
            [0, 1, 0],
            [0, 0.75, 1],
            [0, 0.25, 1],
            [0.25, 0, 1.],
            [1, 0, 1],
            [1, 0, 0.25],
        ])
        elif x.shape[0] == 12:
            mat_weight = np.array([
                [1, 0.25, 0],
                [0.75, 1, 0],
                [0, 2, 0],
                [0, 0.75, 1],
                [0, 0.25, 1],
                [0.25, 0, 1.],
                [1, 0, 1],
                [1, 0, 0.25],
                [1, 0, 0],
                [0, 0, 1],
                [0, 0, 1],
                [1, 0, 0]
            ])
        else:
            print('Wrong shape for this coloring function')
            return (255, 255, 255, 10)

        color = x @ mat_weight

        blend = 0.8  
        thresh = 0.1   
        thresh_max = blend * np.max(color)

        color = np.clip(color, thresh, thresh_max)
        color -= thresh
        color /= thresh_max
        color = np.nan_to_num(color)

        if color.any() and np.linalg.norm(color-np.ones(3))>0.1: #0.35:
            color *=255
            return (color[0], color[1], color[2], 255)       
        else:
            return (255, 255, 255, 10)
    
    def IDstim(self, s):
        ''' Function to convert stim ID from Naumann lab experiment into
            the 8 cardinal directions they correspond to.
        ''' 
        stim = -10
        if s == 3:
            stim = 0
        elif s==10:
            stim = 1
        elif s==9:
            stim = 2
        elif s==16:
            stim = 3
        elif s==4:
            stim = 4
        elif s==14:
            stim = 5
        elif s==13:
            stim = 6
        elif s==12:
            stim = 7
        # add'l stim for specific coloring
        elif s==5:
            stim = 8
        elif s==6:
            stim = 9
        elif s==7:
            stim = 10
        elif s==8:
            stim = 11

        return stim
