import serial #Encapsulates the access for the serial port.
import pyvesc #Allow to communicate with VESC motor controller.
import threading #Allow multithreading
from getch import getch #Getch = getChar, does single-char input.

#Initialising Arduino
arduino = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=1) #All numbers are subjected to change

#6 Motor Functions
def motor1(setRPM):
    ser = serial.Serial('/dev/ttyACM1') #depends on serial on USB hub, Baud rate = 115200
    my_msg = pyvesc.SetRPM(setRPM)
    my_packet = pyvesc.encode(my_msg)
    ser.write(my_packet)
 
def motor2(setRPM):
    ser = serial.Serial('/dev/ttyACM2') #depends on serial on USB hub, Baud rate = 115200
    my_msg = pyvesc.SetRPM(setRPM)
    my_packet = pyvesc.encode(my_msg)
    ser.write(my_packet)
    
def motor3(setRPM):
    ser = serial.Serial('/dev/ttyACM3') #depends on serial on USB hub, Baud rate = 115200
    my_msg = pyvesc.SetRPM(setRPM)
    my_packet = pyvesc.encode(my_msg)
    ser.write(my_packet)
    
def motor4(setRPM):
    ser = serial.Serial('/dev/ttyACM4') #depends on serial on USB hub, Baud rate = 115200
    my_msg = pyvesc.SetRPM(setRPM)
    my_packet = pyvesc.encode(my_msg)
    ser.write(my_packet)
    
def motor5(setRPM): #Vertical Thruster
    ser = serial.Serial('/dev/ttyACM5') #depends on serial on USB hub, Baud rate = 115200
    my_msg = pyvesc.SetRPM(setRPM)
    my_packet = pyvesc.encode(my_msg)
    ser.write(my_packet)
    
def motor6(setRPM): #Vertical Thruster
    ser = serial.Serial('/dev/ttyACM6') #depends on serial on USB hub, Baud rate = 115200
    my_msg = pyvesc.SetRPM(setRPM)
    my_packet = pyvesc.encode(my_msg)
    ser.write(my_packet)

#Multithreading
m1 = threading.Thread(target = motor1) 
m2 = threading.Thread(target = motor2)
m3 = threading.Thread(target = motor3)
m4 = threading.Thread(target = motor4)
m5 = threading.Thread(target = motor5)
m6 = threading.Thread(target = motor6)

m1.start()
m2.start()
m3.start()
m4.start()
m5.start()
m6.start()

#While loop for keyboard command, will be changed for controller or joystick later on
while(True):
    ch = getch() #Wait for keypress
    ch = ch.lower() #change all keypress to lower case
    
    if(ch == 'w'): #forward
        m1 = threading.Thread(target = motor1, arg = 2000)
        m2 = threading.Thread(target = motor2, arg = 2000)
        m3 = threading.Thread(target = motor3, arg = -2000)
        m4 = threading.Thread(target = motor4, arg = -2000)
    elif(ch == 's'): #reverse
        m1 = threading.Thread(target = motor1, arg = -2000)
        m2 = threading.Thread(target = motor2, arg = -2000)
        m3 = threading.Thread(target = motor3, arg = 2000)
        m4 = threading.Thread(target = motor4, arg = 2000)
    elif(ch == 'a'): #left
        m1 = threading.Thread(target = motor1, arg = 2000)
        m2 = threading.Thread(target = motor2, arg = -2000)
        m3 = threading.Thread(target = motor3, arg = 2000)
        m4 = threading.Thread(target = motor4, arg = -2000)
    elif(ch == 'd'): #right
        m1 = threading.Thread(target = motor1, arg = -2000)
        m2 = threading.Thread(target = motor2, arg = 2000)
        m3 = threading.Thread(target = motor3, arg = -2000)
        m4 = threading.Thread(target = motor4, arg = 2000)
    elif(ch == 'q'): #Turn left
        m1 = threading.Thread(target = motor1, arg = 1000)
        m2 = threading.Thread(target = motor2, arg = -1000)
        m3 = threading.Thread(target = motor3, arg = -1000)
        m4 = threading.Thread(target = motor4, arg = 1000)
    elif(ch == 'e'): #Turn right
        m1 = threading.Thread(target = motor1, arg = -1000)
        m2 = threading.Thread(target = motor2, arg = 1000)
        m3 = threading.Thread(target = motor3, arg = 1000)
        m4 = threading.Thread(target = motor4, arg = -1000)
    elif(ch == 'z'): #up
        m5 = threading.Thread(target = motor5, arg = 4000)
        m6 = threading.Thread(target = motor6, arg = 4000)
    elif(ch == 'x'): #down
        m5 = threading.Thread(target = motor5, arg = -2000)
        m6 = threading.Thread(target = motor6, arg = -2000)

    