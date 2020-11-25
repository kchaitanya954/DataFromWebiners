#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 23:08:04 2020

@author: chaitanya
"""
import numpy as np
import matplotlib.pyplot as plt
import librosa as lr
import pafy 
import cv2
import os

class webinar():
    def __init__(self, audio):
        self.audio=audio
        self.amp, self.sr= lr.load(self.audio,res_type='kaiser_fast')
        self.tempo, self.beat_frames=lr.beat.beat_track(y=self.amp, sr=self.sr, \
                                               start_bpm=20)
        
    def innotations(self):
        beat_times = lr.frames_to_time(self.beat_frames, sr=self.sr)
        return len(beat_times)
    
    def silence(self):
        non_silence=0
        non_silence_durations=lr.effects.split(self.amp, frame_length=8000)/self.sr
        for i in range(len(non_silence_durations)):
            non_silence=non_silence+\
                (non_silence_durations[i][1]-non_silence_durations[i][0])
        silence_length=(len(self.amp)/self.sr)-non_silence
        return silence_length/60, non_silence/60
    
    def duration(self):
        return len(self.amp)/(self.sr*60)
    
    def gestures(self,n):
        vid=cv2.VideoCapture(self.audio)
        vid.set(cv2.CAP_PROP_FPS, 10)
        fps = int(vid.get(cv2.CAP_PROP_FPS)) 
        total_frames=vid.get(cv2.CAP_PROP_FRAME_COUNT)        
        vid.set(cv2.CAP_PROP_POS_FRAMES, total_frames)
        check= True
        frame_number = 0
        gestures=[]
        visual=[]
        faces=0
        while check and frame_number <= total_frames:     
            frame_number += n*fps
            vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number)      
            check, frame= vid.read( )    
            if check:            
                gray_img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
                
                fist_cascade=cv2.CascadeClassifier("fist.xml")
                fist=fist_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)
                
                hand_cascade=cv2.CascadeClassifier("haarcascade_hand.xml")
                hand=hand_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)
                
                sign_cascade=cv2.CascadeClassifier("sign_cascade.xml")
                sign=sign_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)
                
                palm_cascade=cv2.CascadeClassifier("palm_v4.xml")
                palm=palm_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)
                
                fpalm_cascade=cv2.CascadeClassifier("closed_frontal_palm.xml")
                fpalm=fpalm_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)
                
                aGest_cascade=cv2.CascadeClassifier("aGest.xml")
                aGest=aGest_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)

                face_cascade= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
                face=face_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5, \
                                   minSize=(10,10))
                if ( len(fist)!=0 or len(hand)!=0 or len(sign)!=0 or len(palm)!=0 \
                    or len(fpalm)!=0 or len(aGest)!=0):
                    gestures.append(True)
                else:
                    gestures.append(False)
                
                if (len(face)!=0):
                    visual.append(True)
                else:
                    visual.append(False)
                    
                if len(face)>faces:
                    faces=len(face)
        gestures_percent= (float(sum(gestures))/float(len(gestures)))*100
        visual_percent=(float(sum(visual))/float(len(visual)))*100
        return gestures_percent, faces, visual_percent

    def graph(self):
        time=np.arange(len(self.amp))/self.sr
        plt.plot(time, self.amp)
        plt.xlabel("Time")
        plt.ylabel("Amptitude")
        plt.show()
        
def statistics(url):
    url=url
    result= pafy.new(url)
    name=result.title
    vid=str(result.streams[0])
    s=vid.index(':')
    e=vid.index('@')
    form=vid[s+1:e]
    result.streams[0].download()
    video_name=name+'.'+form
    speech=webinar(video_name)
    s, ns= speech.silence()
    g, f, v=speech.gestures(60)
    print("Innotations: %d" %(speech.innotations()))
    print("Total Duration: %f" %(speech.duration()) )
    print("Silence duration: %f" %(s))
    print("Non-silence: %f" %(ns))
    print("Percentage of gestures: %f " %(g) )
    print("Number of faces: %f " %(f) )
    print("Visual Duration: %f" %(v*speech.duration()/100))
    print("Visual percentange: %f" %(v))
    os.remove(video_name)
    
if __name__=='__main__':
    url=input("Enter the URL: ")
    statistics(url)
       