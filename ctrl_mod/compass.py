
import serial
import time,threading
import queue
#import codecs
#import struct
#import binascii
from ctypes import *
#from pysabertooth import Sabertooth

ser=serial.Serial('/dev/ttyUSB0',115200)
#myPing = Ping1D()
#myPing.connect_serial("/dev/ttyUSB1",115200)

q = queue.Queue()
q1 = queue.Queue()
start_q = queue.Queue()
#compass code to read data from the compass
def get_angle2():
    #ser.reset_input_buffer()
    #value=0 # random value to make sure that there is no error when the if statement isn't runninbg
    mydata=[]
    #while True:
    count=0
    done =1
    while done !=0:
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
        #return value
        #q.put(round(value,2))
        #print(value)
        round_ang = round(value, 2)
        return round_ang
        #time.sleep(n)#cannot use time sleep


#Sussiest solution ever to prevent the compass from taking a new reference point
#Problem is that compass wont chnage angle depite turning as when turned, it will take that as the new reference point due to the multithreading
def compass_pausable(ref_ang): #This introduces a timer so that compass can take refference and angle per unit time. Hence, allowing us to seee the offset angle
    count = 0
    t1 = time.time()
    while(1):
        
        get_angle2()
            
        t2=time.time()
        #if (count == 0):
            #q.put(get_angle2())
            #count+=1 #exit after getting first value

        if (t2-t1 > 1): #to exit loop after getting as specific time n
            
            q.put(get_angle2())
            break
        
    #start_angle = q.get()#can be used for averaging out
    end_angle = q.get()
    ang_diff = calculate_ang_diff(end_angle, ref_ang)#may need to take out
    q1.put(round(ang_diff, 2)) #can either round off or max 0dp idk up to me or how I feel lmao
    #print("result: ", calculate_ang_diff(start_angle, end_angle))

#Getting diff in angles
def calculate_ang_diff(ang1, ang2): #can use to see whether to turn left or right
    #ang_diff =0
    if ang1 < 0:
        ang1+=360
    if ang2 < 0:
        ang2+=360
        
    if (ang1> ang2):
        ang_diff= ang1-ang2
    elif (ang2 > ang1):
        ang_diff = ang2-ang1
    else:
        ang_diff = 0
        
    return ang_diff

#method to get reference angle
def get_start_angle(timer):
    #timer based? idk
    total_angl = 0
    iteration_counter =0
    t_start = time.time()
    while(1):
        total_angl+=get_angle2()
        iteration_counter +=1
        t_elapsed = time.time()
        #print(get_angle2())
        if(t_elapsed-t_start > timer): #bRuH
            total_angl /= iteration_counter
            ref_angl = round(total_angl, 2)
            start_q.put(ref_angl)
            break
