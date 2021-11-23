import serial
import getopt
from ctypes import *

#Initialise Compass
ser = serial.Serial('/dev/ttyUSB0')

#Getting compass value
    
#showvalue = compassread()

def compassdata():     
    mydata=[]
    count=0
    done =1
    while done !=0:
        for count in range(0,20):
            data = ser.read()
            #'data' returns as unicode code point value of 'data'
            # and converts into hexadecimal value.
            data = hex(ord(data))
            data = int(data,16) #Convert 'data' (hexadecimal) to integer.
            mydata.append(data)
        count=0
        done=0
        
    if (mydata[0] == 0xfa) and (mydata[1] == 0xff):
        #Row
        data1 = mydata[7] << 24
        data2 = mydata[8] << 16
        data3 = mydata[9] << 8
        data4 = mydata[10]
        #Pitch
        data5 = mydata[11] << 24
        data6 = mydata[12] << 16
        data7 = mydata[13] << 8
        data8 = mydata[14]
        #Yaw
        data9 = mydata[15] << 24
        data10 = mydata[16] << 16
        data11 = mydata[17] << 8
        data12 = mydata[18]
        
        #Combining the value to get x,y and z values.
        x = data1 + data2 + data3 + data4
        y = data5 + data6 + data7 + data8
        z = data9 + data10 + data11 + data12
        
        xvalue = pointer(c_int(x)) #Convert 'x' into c_int
        horizontal = cast(xvalue, POINTER(c_float))
        #H=horizontal.contents.value
        
        yvalue = pointer(c_int(y)) #Convert 'y' into c_int
        vertical = cast(yvalue, POINTER(c_float))
        #V=vertical.contents.value
            
        zvalue = pointer(c_int(z)) #Convert 'z' into c_int
        diagonal = cast(zvalue, POINTER(c_float))
        #D=diagonal.contents.value
        
        class compassread:
            H = horizontal.contents.value
            V = vertical.contents.value
            D = diagonal.contents.value
            
            angle1 = str(H)
            angle2 = str(V)
            angle3 = str(D)
            
            value1 = int(H)
            value2 = int(V)
            value3 = int(D)
            
        showvalue = compassread()
        
        #Genie exclusive
        f = open("angletest.txt","a") #Save x value into angle.txt file'
        f.write("x value:")
        f.write(showvalue.angle1)
        f.write("\n")
        f.close()

        #Thorny exclusive
        if (showvalue.value1 > -180) and (showvalue.value1 < 180):
            print(showvalue.value1)

while True:
    compassdata()

    