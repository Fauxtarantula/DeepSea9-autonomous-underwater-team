# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 12:14:44 2021

@author: Rakin
"""
import numpy as np
import time

def arrs():
    t=np.array([1,2,3,4])
    lol = time.time()
    
    return t
def tm():
    seconds=time.time()
    local_time = time.ctime(seconds)
    print(local_time)