# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:23:13 2021

@author: Rakin
"""

#this is to test multithread with diff modules
import threading, queue
import time

q = queue.Queue()



def tryout(n):
    
    x = 3
    q.put(x)
    time.sleep(n)
    
    
#threading.Thread(target=tryout, args=[5]).start()
#print(q.get(1))
#print(q.get(1))