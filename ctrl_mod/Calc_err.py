# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 09:37:12 2021

@author: Rakin
"""

#This functions calculates the error of the angle we are receiving (Current angle) and our desired angle.
def calculate_error_new(ang_set, ang_now, error, value1, value2, value3, value4):
    
    if ang_set == ang_now:
        #Desired angle equals to current angle.
        error = 0
    
    #If desired angle isn't equals to current angle, we need to find the shortest path to travel
    #to get to the desired angle from the current angle.
    elif ang_set > ang_now:
        #Desired angle is larger than current angle.
        #For example, desired angle is 150 and current angle is 90.
        value3 = 360 - ang_set +ang_now # 300 = 360 - 150 + 90
        value4 = ang_set - ang_now #60 = 150 - 90
        
        #Uses the shortest distance here.
        #Since value4<value3, we should use value4's distance.
        if value4 > value3:
            error = value3 #Error becomes value3.
        elif value4 < value3:
            error = value4
        #Incase we have a 180 difference, we need this condition. For example,
        #Desired angle = 180, traveling from 360.
        elif  value4 == value3:
            error = value4
            
    elif ang_set < ang_now:
        value1 = 360 - ang_now +ang_set
        value2 = ang_now - ang_set
        
        if value1 > value2:
            error = value2
        elif value1< value2:
            error = value1
        elif value1 == value2:
            error = value1
            
    return error, value1, value2, value3, value4