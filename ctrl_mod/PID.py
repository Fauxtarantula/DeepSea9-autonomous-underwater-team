# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 15:14:59 2021

@author: Rakin
"""

#to calculate PID
#setpt is the intended angle we needed
#err_last = err_now
#to start timer, we must only activate when angle changed suddenly.
#so, if err_last != err_now, timer starts until angle stop changing

def PID(err_now, err_last, Kp, Ki, Kd, time_s):
    #write the 
    
    #Perr- prop err, Ierr= integral err, Derr= Derivative err
    #notr err last must be global in main code so that 
    dt = time_s
    
    percent_err_n = ((err_now/360)*100)
    percent_err_l =err_last
    
    err_diff = percent_err_n-percent_err_l
    acc_err = percent_err_n + percent_err_l
    P = Kp * percent_err_n
    I = Ki * acc_err  * dt
    D = Kd * (err_diff/dt)
    PID_val = P+I+D
    
    return PID_val, percent_err_n
    #Ierr = 
