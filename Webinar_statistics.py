#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 23:08:04 2020

@author: chaitanya
"""
# Import required packages

import numpy as np 
import matplotlib.pyplot as plt

# !pip install librosa
import librosa as lr  

 # !pip install pafy 
 # To retrieve information such as  viewcount, duration, rating, author, thumbnail, keywords and download videos 
 # !pip install youtube_dl
 # Command-line program to download videos from YouTube.com and other video sites, works with pafy.
import pafy

# !pip install opencv-python
import cv2
import os

# !pip install pyloudnorm
# Flexible audio loudness meter in Python.
import pyloudnorm as pyln


os.chdir("/home/chaitanya/Documents/Webinar parameters")

# To ignore warning as audioread is used instead of Pysoundfile with some formats.
import warnings
warnings.filterwarnings('ignore')

class webinar():

    def __init__(self, video):
        '''
        Used to retrieve Non-verbal parameters such as tempo, loudness, hand gestures, no. of changes in intonations,
        silence duration, visual presence of a human from a given video.
          
        Parameters:
          
        Video: Video file path of most of formats avaliable.
          
        '''
        self.video=video
        self.amp, self.sr= lr.load(self.video,res_type='kaiser_fast')
        self.tempo, self.beat_frames=lr.beat.beat_track(y=self.amp, sr=self.sr, \
                                                start_bpm=10)
    
    
    def innotations(self):
        '''
        Change in intonations are counted, when there is rise or fall in the beats.
          
        caluclate the estimated beat intervals. By caluclating the length of beat intervals.
        We can find out number of times there are changes in intonations.
          
        Return number of changes in intonations.
        '''
        beat_times = lr.frames_to_time(self.beat_frames, sr=self.sr)
        
        return len(beat_times)  
    
    
    def tempo(self):
        '''
        Retuens the estimated global tempo (in beats per minute).
        '''
        return self.tempo
    
    
    def loudness(self):
        '''
        It initially creates BS.1770 meter.
          
        The BS. 1770 measurement is an “integrated” or “overall” measurement - a single loudness value for an\
        entire audio. An integrated meter keeps measuring until you stop it.
          
        '''
        # create BS.1770 meter
        meter = pyln.Meter(self.sr) 
        
        # measure loudness 
        loudness = meter.integrated_loudness(self.amp)
        
        return loudness
    
    def silence(self):
        '''
        Caluclates the non-silence intervals.
        The interval is considered as silent when the sound is less than 30 db.
          
        Retuns the silence and non silence duration in minutes.
        '''
        non_silence=0
        
        #Returns the non silence intervals where sound is more than 30db
        non_silence_durations=lr.effects.split(self.amp, top_db=30, frame_length=2048)/self.sr
        
        for i in range(len(non_silence_durations)):
            non_silence=non_silence+\
                (non_silence_durations[i][1]-non_silence_durations[i][0])
        
        # silence duration is caluclated by subtracting non_silence duration of total duration.       
        silence_length=(len(self.amp)/self.sr)-non_silence
        
        return silence_length/60, non_silence/60
    
    def duration(self):
        '''
        Returns total duration of a video in minutes.
        '''
        return len(self.amp)/(self.sr*60)
    
    def gestures(self,n):
        '''
        Parameters:
        n: The duration between each frame to be captured and analysed in seconds.
          
        For long videos n to be considered more than 10 for better computational speed.
        If n=10, means for every 10 seconds, a frame is captured and analysed.
          
        Returns:
          
        Hand Gestures: Caluclates the percentage of hand gestures observed in the video.
          
        Max faces: Maximum number of faces seen is a frame, is used to find out total number of audience.
          
        Visuals: Caluclates the percentage of duration where a human face is visually present in video.
          
        All these parameters vary based on the selected 'n'.
          
        *** Have the haarcasacde files in the same path . ***
          
        '''
        # Captures the video.
        vid=cv2.VideoCapture(self.video)
        
        # Frames per second is set to 10.
        vid.set(cv2.CAP_PROP_FPS, 10)
        fps = int(vid.get(cv2.CAP_PROP_FPS)) 
        
        # Total number of frames are caluclates.
        total_frames=vid.get(cv2.CAP_PROP_FRAME_COUNT)        
        vid.set(cv2.CAP_PROP_POS_FRAMES, total_frames)
        check= True
        frame_number = 0
        gestures=[]
        visual=[]
        faces=0

        # Loop to read frames in the video
        while check and frame_number <= total_frames: 
        
            # Reads each frame between n seconds.    
            frame_number += n*fps
            vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number)      
            check, frame= vid.read( )    
        
            if check:        
                # Image is converted into B&W for better results.    
                gray_img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        
                
                #hand_cascade=cv2.CascadeClassifier("haarcascade_hand.xml")
                #hand=hand_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)
                
                
                palm_cascade=cv2.CascadeClassifier("palm_v4.xml")
                palm=palm_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)
                
                
                # Identifies hand gestures in a frame
                aGest_cascade=cv2.CascadeClassifier("aGest.xml")
                aGest=aGest_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)
        
                # Used to identify human faces in a frame.
                face_cascade= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
                face=face_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5, \
                                   minSize=(10,10))
                
                # If hand gestures are identifies in a frames then appends True to the grstures list.
                if ( len(palm)!=0 or len(aGest)!=0):
                    gestures.append(True)
                else:
                    gestures.append(False)
                
                # If faces are identifies in a frames then appends True to the faces list.
                if (len(face)!=0):
                    visual.append(True)
                else:
                    visual.append(False)
        
                # Identifies maximum number of faces observed in total frames captured.    
                if len(face)>faces:
                    faces=len(face)
        
        # Caluclates percentage of gestures.
        gestures_percent= (float(sum(gestures))/float(len(gestures)))*100
        
        # Caluclates visual duration of humans.
        visual_percent=(float(sum(visual))/float(len(visual)))*100
        
        return gestures_percent, faces, visual_percent
    
    def graph(self):
        '''
        Returns the amphitude plot of the audio
        '''
        time=np.arange(len(self.amp))/self.sr
        plt.plot(time, self.amp)
        plt.xlabel("Time")
        plt.ylabel("Amptitude")
        plt.show()
        
        
# Method to extract nonverbal features using url.      
# Method to extract nonverbal features using url.      
def statistics_youtube(url, quality=False, n=10, remove= True):

    '''
    It takes youtube link for the video.
    Parameters:
    url: Youtube link of the webinar or any video.
    
    n: The duration between each frame to be captured and analysed in seconds.
    initial is it is set to n=10.
    
    Quality: Initially set as False, If True, downloads the best quality video avaliable, 
    else downloads the lowest quality video for data saving and faster compututions.
    
    Remove: Intitially set True, deletes the downloaded video after extracting the features.
    If False, does nothing.
    
    Prints non verbal parameters such as  and the source video rating given by the viewers. 
    '''
    # reads the youtube video source link
    video= pafy.new(url)
    
    # extracts the name of the video
    name=video.title
    
    # Downloads the best quality video avaliable
    if quality:
        best = video.getbest()
        best.download(quiet=False)
    
        # Gets the extension of the best quality video downloaded
        extension = best.extension
    
    else:
    # Get the lowest quality video avaliable
        video.streams[0].download()
    
        vid=str(video.streams[0])
    
        # Gets the extension of the lowest quality video downloaded
        s = vid.index(':')
        e = vid.index('@')
        extension = vid[s+1:e]
    
    video_name = name+'.'+ extension
    
    # assigning webinar class for a given video to a object speech.
    speech=webinar(video_name)
    
    # silence and non silence durations are calucalted
    s, ns= speech.silence()
    
    # Percentage of gestures, visual duration and maximun number of faces.
    g, f, v=speech.gestures(n)
    print()
    # Prints the number of changes in intonations in a speech.
    print("Innotations: {}" .format(speech.innotations()))
    
    # Prints the loudness of the whole speech.
    print("Loudness: {}" .format(speech.loudness))
    
    # Prints the tempo of the speech
    print("Tempo: {}" .format(speech.tempo))
    
    # Prints the total duration of the video.
    print("Total Duration: {}" .format(speech.duration()) )
    
    # Prints the silence duration in the speech.
    print("Silence duration: {}" .format(s))
    
    # Prints the non-silence duration in the speech.
    print("Non-silence: {}" .format(ns))
    
    # Prints the the percentage of gestures in a video.
    print("Percentage of gestures: {}" .format(g) )
    
    # Prints the maximum number of human faces captures in a frame.
    print("Number of faces: {}" .format(f) )
    
    # Prints the total duration of human face visual presence in a video.
    print("Visual Duration: {}" .format(v*speech.duration()/100))
    
    # Prints the percentage of human face visual presnece in a video.
    print("Visual percentange: {}" .format(v))
    
    # Prints the rating of the video 
    print("Rating: {}" .format(video.rating))
    
    # Prints the likes of the video
    print("Likes: {}" .format(video.likes))
    
    # Prints the dislikes of the video
    print("Dislikes: {}" .format(video.dislikes))
    
    # Removes the video from the source after extracting the features.
    if remove:
        os.remove(video_name)
        
# Method to extract non verbal parameters using videos source path.
def statistics_system(path, n=10):

    '''
    Parameters:
    path: Path of the video file.
    
    n: The duration between each frame to be captured and analysed in seconds.
    initial is it is set to n=10.
    
    Prints non verbal parameters such as  and the source video rating given by the viewers.
    
    '''
    # assigning webinar class for a given video to a object speech.
    speech=webinar(path)
    # silence and non silence durations are calucalted
    s, ns= speech.silence()
    
    # Percentage of gestures, visual duration and maximun number of faces.
    g, f, v=speech.gestures(n)
    print()
    # Prints the number of changes in intonations in a speech.
    print("Innotations: {}" .format(speech.innotations()))
    
    # Prints the loudness of the whole speech.
    print("Loudness: {}" .format(speech.loudness))
    
    # Prints the tempo of the speech
    print("Tempo: {}" .format(speech.tempo))
    
    # Prints the total duration of the video.
    print("Total Duration: {}" .format(speech.duration()) )
    
    # Prints the silence duration in the speech.
    print("Silence duration: {}" .format(s))
    
    # Prints the non-silence duration in the speech.
    print("Non-silence: {}" .format(ns))
    
    # Prints the the percentage of gestures in a video.
    print("Percentage of gestures: {} " .format(g) )
    
    # Prints the maximum number of human faces captures in a frame.
    print("Number of faces: {} " .format(f) )
    
    # Prints the total duration of human face visual presence in a video.
    print("Visual Duration: {}" .format(v*speech.duration()/100))
    
    # Prints the percentage of human face visual presnece in a video.
    print("Visual percentange: {}" .format(v))


    
if __name__=='__main__':
    url=input("Enter the URL: ")
    statistics_youtube(url)
       
