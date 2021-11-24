import serial
import getopt
import pyvesc
from ctypes import *

'''
If the Z axis function is activated, it will fight with you for the direction of the HUV.

The purpose of the Z axis is for the robot to square to the front.

The starting angle of the Z axis has varied everytime we plugged in.
'''
            
def zmaintain(): #Remember to check where the starting point for z axis is
    if(showvalue.value3 <= 170 and showvalue.value3 >= 0): #Z axis, Adjust to the right
        motor1(-1000)
        motor2(1000)
        motor3(1000)
        motor4(-1000)
    elif(showvalue.value3 >= -190 and showvalue.value3 <= -20): #Z axis, Adjust to the left
        motor1(1000)
        motor2(-1000)
        motor3(-1000)
        motor4(1000)
    elif(showvalue.value3 < 0 and showvalue.value3 > -20): #Z axis, Do nothing
        motor1(0)
        motor2(0)
        motor3(0)
        motor4(0)