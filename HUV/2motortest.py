import serial #Encapsulates the access for the serial port.
import pyvesc #Allow to communicate with VESC motor controller.
from getch import getch #Getch = getChar, does single-char input.

'''
Notes:
    min RPM = 900
    max RPM = 9000
    +ve, forward
    -ve, backward
    
Code description:
    When code is run(Ctrl + T), holding 'd' will move the motor forward,
    like wise holding 'a' will move the motor backwards.
    
Make sure both ESCs are plugged in before running the code!
'''


def motor1(setRPM):
    ser = serial.Serial('/dev/ttyACM0') #ACM0, USB port 0
    my_msg = pyvesc.SetRPM(setRPM)
    my_packet = pyvesc.encode(my_msg)
    ser.write(my_packet)
    
def motor2(setRPM):
    ser = serial.Serial('/dev/ttyACM1') #ACM0, USB port 1
    my_msg = pyvesc.SetRPM(setRPM)
    my_packet = pyvesc.encode(my_msg)
    ser.write(my_packet)

while(True):
    ch = getch() #Wait for keypress
    ch = ch.lower() #change all keypress to lower case
    if(ch == 'a'):
        motor1(2000) #Forward
        motor2(0)
    elif (ch == 'd'):
        motor1(0)
        motor2(2000)
        
        
        
        