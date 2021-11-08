# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 09:20:54 2021

@author: Rakin
"""

import serial
import time #Time functions
import cv2
import numpy as np #Using numpy for array function
from ctypes import *
import math

#to call module -> import Get_Ang_Mod as angl
                  # angl.get_angle(parameter = serial_data)
                  
def get_angle(ser):
    global value
    ser.reset_input_buffer()
    mydata=[]
    while True:
        count=0
        done =1
        while done !=0:
            for count in range(0,20):
                data = ser.read()
                data=hex(ord(data)) 
                data=int(data,16)
                mydata.append(data)
            count=0
            done=0
        
        #Checks whether compass is receiving data.
        if (mydata[0]==0xfa) and (mydata[1] ==0xff):
            #Shifting bytes to make 3 principle axes. X (Roll), Y (Pitch) and Z(Yaw)
            data1=mydata[7]<<24
            data2=mydata[8]<<16
            data3=mydata[9]<<8
            data4=mydata[10]
            data5=mydata[11]<<24
            data6=mydata[12]<<16
            data7=mydata[13]<<8
            data8=mydata[14]
            data9=mydata[15]<<24
            data10=mydata[16]<<16
            data11=mydata[17]<<8
            data12=mydata[18]
         
            x = data1 + data2 + data3 + data4 #Roll
            y= data5 + data6 + data7 + data8 #Pitch
            z = data9 + data10 + data11 + data12 #Yaw
                
            cp = pointer(c_int(z)) 
            fp = cast(cp,POINTER(c_float))
            value = fp.contents.value
           
        if (-180<value) and (value<180):
            return value 