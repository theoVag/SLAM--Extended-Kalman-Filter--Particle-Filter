#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 19:34:35 2019

@author: 
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from filterpy.stats import plot_covariance_ellipse
from time import sleep
class Kalman:
    
    def __init__(self,x,Q,R,P=None):
        
        self.x= x
        self.x[2] = calc_angle(self.x[2])
        
        self.Q = Q
        
        self.R=R
        if P is None:
            self.P= np.eye(len(x))
            
        else:
            self.P= P
        
        self.landmarks_first_seen=0
    def predict(self,ut,w,dt):
        
        self.F=calc_jacobian(0,ut,w,dt,self.x)
        
        x =np.asscalar(self.x[0])
        y =np.asscalar(self.x[1])
        theta = np.asscalar(self.x[2])
        xob1 = np.asscalar(self.x[3])
        yob1 =np.asscalar(self.x[4])
        xob2 =np.asscalar(self.x[5])
        yob2 =np.asscalar(self.x[6])
        
        self.x = np.matrix([[x+np.cos(theta)*ut*dt],[y+np.sin(theta)*ut*dt],[theta + w*dt],[xob1],[yob1],[xob2],[yob2]])
        self.x[2] = calc_angle(self.x[2])

        self.P = self.F*self.P*np.transpose(self.F) + self.Q 
        
    
    def update(self,ut,w,dt,Z):
        
        if (self.landmarks_first_seen==0):
            d1= Z[0]
            phi1= Z[1]
            d2= Z[2]
            phi2= Z[3]
            self.x[3] = self.x[0] + d1*np.cos(phi1+self.x[2])
            self.x[4] = self.x[1] + d1*np.sin(phi1+self.x[2])
            self.x[5] = self.x[0] + d2*np.cos(phi2+self.x[2])
            self.x[6] = self.x[1] + d2*np.sin(phi2+self.x[2])
            self.landmarks_first_seen=1
        
        
        self.H=calc_jacobian(1,ut,w,dt,self.x)
        
        x =np.asscalar(self.x[0])
        y =np.asscalar(self.x[1])
        theta = np.asscalar(self.x[2])
        xob1 = np.asscalar(self.x[3])
        yob1 =np.asscalar(self.x[4])
        xob2 =np.asscalar(self.x[5])
        yob2 =np.asscalar(self.x[6])
        
        hz= np.matrix([[np.sqrt((x-xob1)**2+(y-yob1)**2)],
                        [calc_angle(np.arctan2(yob1-y,xob1-x))-theta],
                        [np.sqrt((x-xob2)**2+(y-yob2)**2)],
                        [calc_angle(np.arctan2(yob2-y,xob2-x))-theta]])       
          

        self.Y = Z - hz
        self.Y[1] = calc_angle(self.Y[1])
        self.Y[3] = calc_angle(self.Y[3])

        self.S = self.H*self.P*np.transpose(self.H) + self.R
        self.K= self.P*np.transpose(self.H)*np.linalg.inv(self.S)
        self.x = self.x + self.K*self.Y
        #self.x = np.matrix(self.x).astype(float)
        self.x[2] = calc_angle(self.x[2])

        temp = self.K*self.H
        self.P = (np.eye(temp.shape[0])-temp)* self.P
        
def calc_jacobian(order,ut,w,dt,xs):
    x =np.asscalar(xs[0])
    y =np.asscalar(xs[1])
    th=np.asscalar(xs[2])
    xob1 = np.asscalar(xs[3])
    yob1 =np.asscalar(xs[4])
    xob2 =np.asscalar(xs[5])
    yob2 =np.asscalar(xs[6])
    
    if(order==0): # F jacobian of the motion model
        
        J= np.matrix(([[1, 0, -dt*ut*np.sin(th), 0, 0, 0, 0],
                       [0, 1,  dt*ut*np.cos(th), 0, 0, 0, 0],
                       [0, 0,1, 0, 0, 0, 0],
                       [0, 0,0, 1, 0, 0, 0],
                       [0, 0,0, 0, 1, 0, 0],
                       [0, 0,0, 0, 0, 1, 0],
                       [0, 0,0, 0, 0, 0, 1]])).astype(np.float64)
    else: # H jacobian of the measurements
        
        q = np.sqrt((x - xob1)**2 + (y - yob1)**2)
        q2 = np.sqrt((x - xob2)**2 + (y - yob2)**2)
        J=np.matrix([[(x - xob1)/q, (y - yob1)/q,  0, (-x + xob1)/q, (-y + yob1)/q,0,0],
        [ -(y - yob1)/(q**2), -(-x + xob1)/(q**2), -1,(y - yob1)/(q**2),(-x + xob1)/(q**2),0, 0],
        [(x - xob2)/q2, (y - yob2)/q2,  0,0,0, (-x + xob2)/q2, (-y + yob2)/q2],
        [ -(y - yob2)/(q2**2), -(-x + xob2)/(q2**2), -1,0,0,(y - yob2)/(q2**2),(-x + xob2)/(q2**2)]]).astype(np.float64)
    
    return J


def calc_angle(a):
    a = a % (2 * np.pi)
    
    if a > np.pi:             # move to [-pi, pi)
        a -= 2 * np.pi
        
    return a

if __name__ == "__main__":

    control1 = pd.read_csv("dataset/control1.csv", header=None)
    radar1 = pd.read_csv("dataset/radar1.csv", header=None)

    for i in range(0,radar1.shape[0]):
        radar1.iloc[i,1] = calc_angle(radar1.iloc[i,1])
        radar1.iloc[i,3] = calc_angle(radar1.iloc[i,3])
        #pass

    dt=0.1
    # Calculate initial values-----------------------------------------------------------edw gt pio meta?
    init_val1=np.zeros([2,1])
    init_val2=np.zeros([2,1])
    d1= radar1.iloc[0,0]
    phi1= radar1.iloc[0,1]
    d2= radar1.iloc[0,2]
    phi2= radar1.iloc[0,3]
    init_val1[0] = 0 + d1*np.cos(phi1)
    init_val1[1] = 0 + d1*np.sin(phi1)
    init_val2[0] = 0 + d2*np.cos(phi2)
    init_val2[1] = 0 + d2*np.sin(phi2)

    #state = np.matrix([[0],[0],[0],init_val1[0],init_val1[1],init_val2[0],init_val2[1]]).astype(np.float64)
    state = np.matrix([[0],[0],[0],[0],[0],[0],[0]]).astype(np.float64)
    # exoume u=1 ews 2.kai me max u*1*0.1 = 0.2 kai an upothesoume thorubo 5-10% tha exoume thorubo std 0.01-0.02 kai meta sto tetragwno
    Q = np.matrix([[0.01**2, 0.0, 0.0, 0.0, 0.0, 0.0 , 0.0], 
                   [0.0, 0.01**2, 0.0, 0.0, 0.0, 0.0, 0.0], 
                   [0.0, 0.0, 0.01**2 , 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0,0.001**2, 0.0, 0.0, 0.0],
                   [0.0, 0.0 , 0.0, 0.0,0.001**2, 0.0, 0.0],
                   [0.0, 0.0 , 0.0, 0.0, 0.0,0.001**2, 0.0],
                   [0.0, 0.0,0.0, 0.0, 0.0, 0.0,0.001**2]])

    R = np.matrix([[0.25, 0.0, 0.0,0.0], [0.0, 0.09, 0.0,0.0], 
                   [0.0, 0.0, 0.25,0.0],[0.0, 0.0, 0.0,0.09]])
    
    P= np.matrix([[3, 0.0, 0.0, 0.0, 0.0, 0.0 , 0.0], 
                  [0.0, 3, 0.0, 0.0, 0.0, 0.0, 0.0], 
                  [0.0, 0.0, np.deg2rad(10) , 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0,10, 0.0, 0.0, 0.0],
                  [0.0, 0.0 , 0.0, 0.0,10, 0.0, 0.0],
                  [0.0, 0.0 , 0.0, 0.0, 0.0,10, 0.0],
                  [0.0, 0.0,0.0, 0.0, 0.0, 0.0,10]])
    
    kalman_filter = Kalman(state,Q,R,P)
    output_listx=[]
    output_listy=[]
    plt.ion()
    results=[]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot([], [])

    for t in range(0,radar1.shape[0]):
        u = control1.iloc[t,0]
        wmega = control1.iloc[t,1]
        
        tempZ= np.matrix([[radar1.iloc[t,0]],[radar1.iloc[t,1]],[radar1.iloc[t,2]],[radar1.iloc[t,3]]])
        
        kalman_filter.predict(u,wmega,dt)
        
        kalman_filter.update(u,wmega,dt,tempZ)
        
        output = kalman_filter.x
        results.append(output)
        output_listx.append(np.asscalar(output[0]))
        output_listy.append(np.asscalar(output[1]))
        line1.set_ydata(output_listy)
        line1.set_xdata(output_listx)

        lambda_, v = np.linalg.eig(kalman_filter.P[0:2,0:2])
        lambda_ = np.sqrt(lambda_)

        plot_covariance_ellipse(np.array([(output[5]), (output[6])]), kalman_filter.P[5:7,5:7], facecolor='lightgreen', alpha=0.1)
        plot_covariance_ellipse(np.array([(output[3]), (output[4])]), kalman_filter.P[3:5,3:5], facecolor='lightcoral', alpha=0.1)
        plot_covariance_ellipse(np.array([(output[0]), (output[1])]), kalman_filter.P[0:2,0:2], facecolor='lightskyblue', alpha=0.1)
        
        ax.scatter(np.asscalar(output[3]),np.asscalar(output[4]),color='r',zorder=t)
        ax.scatter(np.asscalar(output[5]),np.asscalar(output[6]),color='g',zorder=t)
        ax.set_title('Kalman filter iter=%d'%(t+1))
        ax.relim()
        ax.autoscale_view()
        #print(lambda_[0], lambda_[1])
        #print(kalman_filter.P[0:2,0:2])
        sleep(0.05)
        fig.canvas.draw()
        fig.canvas.flush_events()
        #print(t)
        
    print(output)