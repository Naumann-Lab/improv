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
        self.num_stim = 12 # change this to include all the monocular stims
        self.frame = 0
        # self.curr_stim = 0 #start with zeroth stim unless signaled otherwise
        self.stim = {}
        self.stimStart = -1
        self.currentStim = None
        self.ests = np.zeros((1, self.num_stim, 2)) #number of neurons, number of stim, on and baseline
        self.counter = np.ones((self.num_stim,2)) # stim counter?
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
        np.savetxt('output/analysis_proc_S.txt', np.array(self.S))
        
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
            ids = self.q_in.get(timeout=0.0001) # from processor actor 
            if ids is not None and ids[0]==1:
                print('analysis: missing frame')
                self.total_times.append(time.time()-t)
                self.q_out.put([1])
                raise Empty

            # collecting neuronal activity data
            self.frame = ids[-1]
            (self.coordDict, self.image, self.C) = self.client.getList(ids[:-1])
            self.C = np.where(np.isnan(self.C), 0, self.C)

            if self.coordDict is not None:
                self.coords = [o['coordinates'] for o in self.coordDict]
            
            # Compute tuning curves based on input stimulus
            # Just do overall average activity for now
            try: 
                ## stim format: stim, stimonOff, angle, vel, freq, contrast

                # how am I getting the input_stim_queue from the analysis actor, when this is the analysis actor?
                sig = self.links['input_stim_queue'].get(timeout=0.0001) 
                self.updateStim_start(sig)
                self.stimText = list(sig.values())
            except Empty as e:
                pass #no change in input stimulus

            self.stimAvg_start()
            
            self.globalAvg = np.mean(self.estsAvg[:,:8], axis=0)
            self.tune = [self.estsAvg[:,:8], self.globalAvg] # what does this tune part look like? and this is global?

            # Compute coloring of neurons for processed frame
            # Also rotate and stack as needed for plotting
            self.color, self.tc_list = self.plotColorFrame()
            
            ## TODO: Add in my analysis function here...I want to keep all the coloring for visualization 

            # identify the tuning colors with a ID (i.e. forward for most green)
            # remap to dictionary? just do it without a for loop

            # or make a new analysis function that is the unbiased clustering analysis?

            # can i make a new analysis that only will run once the stimulus is done and before I want to start anything else?
            # maybe connect this to the gui with an 'analyze' button - would run the unbiased clustering algorithm that I have already made
            
            # or pause improv after I collect the functional information and then start it again?

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

            self.putAnalysis() # sends out the analysis info here

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

        # assuming we have one of those 8 stimuli
        if stimID != -10:
            if stimID not in self.allStims.keys():
                self.allStims.update({stimID:[]})
            # determine if this is a new stimulus trial
            if abs(stim[frame][1])>1 :
                curStim = 1 #on
                self.allStims[stimID].append(frame)
            else:
                curStim = 0 #off
            # paradigm for these trials is for each stim: [off, on, off]
            if self.lastOnOff is None:
                self.lastOnOff = curStim
            elif curStim == 1: #self.lastOnOff == 0 and curStim == 1: #was off, now on
                # select this frame as the starting point of the new trial
                # and stimulus has started to be shown
                # All other computations will reference this point
                self.stimStart = frame 
                self.currentStim = stimID
                if stimID<8:
                    self.currStimID[stimID, frame] = 1
                # NOTE: this overwrites historical info on past trials
                logger.info('Stim {} started at {}'.format(stimID,frame))
            else:
                self.currStimID[:, frame] = np.zeros(8)
            print('Frame: ', frame, 'On off :', self.lastOnOff)
            print('Current data frame is ', self.frame)

    def putAnalysis(self):
        ''' Throw things to DS and put IDs in queue for Visual
        '''
        t = time.time()
        ids = []
        ids.append(self.client.put(self.Cx, 'Cx'+str(self.frame))) #x axis of time
        ids.append(self.client.put(self.Call, 'Call'+str(self.frame))) # ??
        ids.append(self.client.put(self.Cpop, 'Cpop'+str(self.frame))) # population activity
        ids.append(self.client.put(self.tune, 'tune'+str(self.frame))) # tuning curves
        ids.append(self.client.put(self.color, 'color'+str(self.frame))) # color of each neuron
        ids.append(self.client.put(self.coordDict, 'analys_coords'+str(self.frame))) # coordinates of each neuron
        ids.append(self.client.put(self.allStims, 'stim'+str(self.frame))) # all the tested stimuli
        ids.append(self.client.put(self.tc_list, 'tc_list')) # tuning curve list?
        ids.append(self.frame)

        self.q_out.put(ids)
        self.puttime.append(time.time()-t)

    def stimAvg_start(self):
        '''
        Computing the average cell activity during the stimulus on portion
        '''

        t = time.time()

        ests = self.C # estimates
        
        if self.ests.shape[0]<ests.shape[0]:
            diff = ests.shape[0] - self.ests.shape[0]
            # added more neurons, grow the array
            self.ests = np.pad(self.ests, ((0,diff),(0,0),(0,0)), mode='constant')
           
        before_amount = 5
        after_amount = 20

        ## TODO: what exactly is being saved here?

        if self.currentStim is not None:
            if self.stimStart == self.frame:
                # account for the baseline prior to stimulus onset
                mean_val = np.mean(ests[:,self.frame-before_amount:self.frame],1) #baseline mean value

                # normalized to baseline response for the current stimulus?? unsure here
                self.ests[:,self.currentStim,1] = (self.counter[self.currentStim,1]*self.ests[:,self.currentStim,1] + 
                                                   mean_val) / (self.counter[self.currentStim,1]+1)
                self.counter[self.currentStim, 1] += before_amount # why add a couple more?
                    
            # if the image frame is the next frame post stimulus onset, then add the value from that frame
            elif self.frame in range(self.stimStart+1, self.stimStart+2):
                val = ests[:,self.frame-1]
                self.ests[:,self.currentStim,1] = (self.counter[self.currentStim,1]*self.ests[:,self.currentStim,1] + 
                                                   val)/(self.counter[self.currentStim,1]+1)
                self.counter[self.currentStim, 1] += 1

            # if the image frame is the next couple frames post stimulus onset, then add the value from that frame
            elif self.frame in range(self.stimStart+2, self.stimStart+after_amount):
                val = ests[:,self.frame-1]
                self.ests[:,self.currentStim,0] = (self.counter[self.currentStim,0]*self.ests[:,self.currentStim,0] + 
                                                   val)/(self.counter[self.currentStim,0]+1)
                self.counter[self.currentStim, 0] += 1

        self.estsAvg = np.squeeze(self.ests[:,:,0] - self.ests[:,:,1])        
        self.estsAvg = np.where(np.isnan(self.estsAvg), 0, self.estsAvg)
        self.estsAvg[self.estsAvg == np.inf] = 0
        self.estsAvg[self.estsAvg<0] = 0
        # now we have an estimates average for all the neurons per stimulus

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
                    logger.error('Error in fill poly: {}'.format(e))
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

        # add in here more color ids for the increased number of stimuli
        # or keep binocular and monocular separate...

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

        # how do these stim numbers correspond??

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
