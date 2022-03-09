#this code is to decide to turn left or right


import math
import queue

queue_dir = queue.Queue()
def direction(err):
    #queue_dir = queue.Queue()
    bol = 0

    if err >1:
        bol = 0
    elif err >-1 and err<1 :
        bol =1
    else:
        bol = 2
    return bol
