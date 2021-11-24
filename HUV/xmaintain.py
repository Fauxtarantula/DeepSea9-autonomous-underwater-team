import serial
import getopt
import pyvesc
from ctypes import *

def xmaintain():    
    if(showvalue.value1 <= 90 and showvalue.value1 >= 10): #X axis, Adjust right tilt, might expect slight descend
        motor5(1000)
        motor6(-1000)
    elif(showvalue.value1 >= -90 and showvalue.value1 <= -10): #X axis, Adjust left tilt, might expect slight descend
        motor5(-1000)
        motor6(1000)
    elif(showvalue.value1 > -10 and showvalue.value1 < 10): #X axis, Depth Maintenence
        motor5(3000)
        motor6(3000)
    else: #Vehicle overturn
        motor5(0)
        motor6(0)