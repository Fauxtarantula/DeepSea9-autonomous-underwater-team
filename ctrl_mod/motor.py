# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 10:01:56 2021

@author: Rakin
"""
import numpy as np
import cv2
import time
#from ctypes import *
#from pysabertooth import Sabertooth #uncomment once doing real testing
saber=Sabertooth('/dev/ttyS0',baudrate=9600,address=128,timeout=0.1)
tre = 3

def FORWARD(m1, m2):
    saber.drive(2,-m1)
    saber.drive(1,-m2)
    
    print("m1:",m1)
    print("m2:",m2)
    print("Moving forward\n")
    #print(tre) #having a local variable in a module can still be called in main code lmao did you forget java rakin jeez
    print("=================")
    
    
def RIGHT_TURN(m1, m2,M):
    new_m1=int(m1*(1.0-(M/100.0)))
    saber.drive(2,-new_m1)
    saber.drive(1,-m2)
    
    print("m1:",new_m1)
    print("m2:",m2)
    print("Turning right")
    print("=================")

def LEFT_TURN(m1,m2,M):
    new_m2=int(m2*(1.0-(M/100.0)))
    saber.drive(2,-m1)
    saber.drive(1,-new_m2)
    
    print("m1:",m1)
    print("m2:",new_m2)
    print("Turning left")
    print("=================")
               
def stop():
    saber.drive(1,0)
    saber.drive(2,0)
    
    print("m1: 0")
    print("m2: 0")
    print("Stopping\n")
    
