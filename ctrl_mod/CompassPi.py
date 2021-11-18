import serial
import time,threading
#import codecs
#import struct
#import binascii
import queue
from ctypes import *
#from pysabertooth import Sabertooth


#ser=serial.Serial('/dev/ttyUSB1',115200)
ser=serial.Serial('/dev/ttyUSB1',115200)
#myPing = Ping1D()
#myPing.connect_serial("/dev/ttyUSB1",115200)

#start queue sequence here to save value
q = queue.Queue()

#compass code to read data from the compass
def get_angle2(n):
    #ser.reset_input_buffer()
    #value=0 # random value to make sure that there is no error when the if statement isn't runninbg
    mydata=[]
    #while True:
    count=0
    done =1

    for count in range(0,20):
        data = ser.read()
            #print(data)
        data=hex(ord(data)) #gives in str
            #print (data)
        data=int(data,16)
        mydata.append(data)
        count=0
        done=0
        #print(mydata[0],mydata[1])
            
if (mydata[0]==0xfa) and (mydata[1] ==0xff):

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
     
    x = data1 + data2 + data3 + data4
    y= data5 + data6 + data7 + data8
    z = data9 + data10 + data11 + data12
            
    cp= pointer(c_int(z)) #makes x into c int
    fp=cast(cp,POINTER(c_float))
    H=fp.contents.value
            
    xp= pointer(c_int(x)) #makes x into c int
    tp=cast(xp,POINTER(c_float))
    X=tp.contents.value
        #X=int(X)
        #print(X)
    angle1=str(X)
    #f=open("angle.txt","a")
    #f.write("x value:")
    #f.write(angle1)
    #f.write("\n")
    #f.close()    
        
    value=H
    #angle=X
    angl = round(value,2)
    q.put(angl)   
       
    #if (-180<H) and (H<180):
        #print(value)
        
    time.sleep(n)
    
#while(1):
    #get_angle2()

