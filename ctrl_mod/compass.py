import serial
import time,threading
import queue
#import codecs
#import struct
#import binascii
from ctypes import *
#from pysabertooth import Sabertooth

ser=serial.Serial('/dev/ttyUSB0', 9600) #115200

ser2=serial.Serial('/dev/ttyACM0', 9600, timeout = 1) #initialize serial with arduino
ser2.write(b"H") #think of better placement
#myPing = Ping1D()
#myPing.connect_serial("/dev/ttyUSB1",115200)

q = queue.Queue()
q1 = queue.Queue()
start_q = queue.Queue() #should probably use 1 queue
desire_q = queue.Queue()

#compass code to read data from the compass
def get_angle2():
    ser.reset_input_buffer()
    ser2.reset_input_buffer()
    ser2.write(b"H")
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

        #try: #try catch block
            #yes = double(value)
            #round_ang = round(value, 2)
            
        #except:
            #round_ang = 0
         
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
    ang_diff, variant = calculate_ang_diff(end_angle, ref_ang)#may need to take out
    q1.put(round(ang_diff, 2))
    q1.put(variant)#can either round off or max 0dp idk up to me or how I feel lmao
    #print("result: ", calculate_ang_diff(start_angle, end_angle))

#Getting diff in angles
def calculate_ang_diff(ang1, ang2): #can use to see whether to turn left or right, need to change abit. ang2 is the ref_ang
        
    if (ang1 > ang2):
        #ang_diff= ang1-ang2
        ang_diff = ang1-ang2
        boo = 0#this is a terrible programming practice. DO NOT EVER FUCKING USE THIS IN INDUSTRY
        
    elif (ang2 > ang1):
        #ang_diff = ang1-ang2
        ang_diff = ang2-ang1
        #ang_diff = abs[ang_diff]
        boo = 1
    else:
        ang_diff = 0
        boo = 2
        
    return ang_diff, boo

#method to get reference angle
def get_start_angle(timer): #should probably give 10-120 seconds
    #timer based? idk
    total_angl = 0
    iteration_counter =0
    t_start = time.time()
    
    while(1):   
        try: #failsafe try-catch block, this is also another hack I created to make sure the compass reads
            total_angl+=get_angle2()
        except:
            #total_angl = 0
            print("---Compass stopped working---Reinitiating again---get_start_angle thread")
            #total_angl += get_angle2()
            
        iteration_counter +=1
        t_elapsed = time.time()
        #print(get_angle2())
        if(t_elapsed-t_start > timer): #bRuH
            total_angl /= iteration_counter
            ref_angl = round(total_angl, 2)
            start_q.put(ref_angl)
            print("Done taking ref angl, 10 seconds starts")
            time.sleep(10)
            break

def get_desired(ref_ang):
    desire_start = time.time()
    total_desire_ang = 0
    i_counter = 0
    
    while(1):
        try:
            total_desire_ang +=get_angle2()
            
        except:
            print("---Compass stopped working---Reinitiating again---get_desired thread")
        
        i_counter+=1
        desire_end =time.time()
        if(desire_end-desire_start > 5):#10 seconds for now
            total_desire_ang /=i_counter
            desire = round(total_desire_ang, 2)
            
            fuck, me = calculate_ang_diff(desire, ref_ang)
            desire_q.put(fuck)
            print("Done taking desire")
            break
#For Testing        
#thread1 = threading.Thread(target = compass_pausable, args = [3])
thread1 = threading.Thread(target = get_start_angle, args = [10])
thread1.start()
#print("1: ",q1.get())
#print("2: ",q.get())

thread1.join()
ref= start_q.get()
thread2 = threading.Thread(target = get_desired, args = [ref])
thread2.start()
thread2.join()
#print(desire_q.get())

desire = desire_q.get()
 #cannot work with while loop
while(1):
#ser2.write(b"H") #must be in bytes
    #print(get_angle2())
    
    #thread1 = threading.Thread(target = get_start_angle, args = [1])
    #thread1.start()
    #print("1: ",start_q.get())
    #thread1.join()
    
    thread3 = threading.Thread(target = compass_pausable, args = [ref])
    thread3.start()
    #print(q1.get())
    thread3.join()
    err = q1.get()#float(round(q1.get(),2))
    variant = q1.get()
    
    
    if variant == 0:# important part
        new_err=desire+err
    else:
        new_err=err-desire
        
        
    print(new_err)
    print(variant)
    
    while not q1.empty(): #flush out
        q1.get()
