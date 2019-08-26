#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:20:22 2019

@author: 
"""

import numpy as np
from numpy.linalg import norm
from numpy.random import randn
from numpy.random import uniform
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
import random
from numpy.random import rand
from scipy.stats import multivariate_normal
from time import sleep
import math
import matplotlib
import time
def calc_angle(a):
    a = a % (2 * np.pi)
    
    #if a > np.pi:             # move to [-pi, pi)
    #    a -= 2 * np.pi
        
    a[a>np.pi] -= 2 * np.pi
    return a
def single_calc_angle(a):
    a = a % (2 * np.pi)
    
    if a > np.pi:             # move to [-pi, pi)
        a -= 2 * np.pi
        
    #a[a>np.pi] -= 2 * np.pi
    return a

def initialize_unif_part(x_range, y_range, theta_range,v_range, N):
    particles = np.empty((N, 5))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(theta_range[0], theta_range[1], size=N)
    
    particles[:, 3] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 4] = uniform(v_range[0], v_range[1], size=N)

    particles[:,2] = calc_angle(particles[:,2])
    
    return particles
    
def create_gaussian_particles(mean, std, N):
    particles = np.empty((N, 5))
    particles[:, 0] = mean[0] + (randn(N) * std[0])
    particles[:, 1] = mean[1] + (randn(N) * std[1])
    particles[:, 2] = mean[2] + (randn(N) * std[2])
    particles[:,2] = calc_angle(particles[:,2])
    
    particles[:, 3] = mean[3] + (randn(N) * std[3])
    particles[:, 4] = mean[4] + (randn(N) * std[4])
    return particles


def normpdf(x, mu, std):

    cov = np.diag((1,1))*std

    part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)) )
    part2 = (-1/2) * ((x-mu).T.dot(np.linalg.inv(cov))).dot((x-mu))
    return float(part1 * np.exp(part2))
   
class ParticleFilter:
    
    def __init__(self,N,landmarks,Q_std,R_std,init_x=None):
        
        self.N = N
        # 4x1 array
        self.landmarks = landmarks
        
        if init_x is None:
            self.particles = initialize_unif_part((-12,12),(-12,12),(-10,10),(-np.pi,np.pi),N)
        else:
            #self.particles = create_gaussian_particles(init_x,Q_std,N)
            self.particles = create_gaussian_particles(init_x,np.array([0.1,0.1,0.01,0.01,0.1]),N)  #(init_x,np.array([0.1,0.1,0.01,0.01,0.1]),N)

        self.weights = np.ones(N)/N
        self.particles[:,2] = calc_angle(self.particles[:,2])
        self.Q_std = Q_std
        self.R_std=R_std
        
    def predict(self,u,dt):
        N=self.N
        std = self.Q_std
        
        
        """self.particles[:,0] = self.particles[:,0] + np.cos(self.particles[:,2])*u[0]*dt + (randn(N)*std[0])
        self.particles[:,1] = self.particles[:,1] + np.sin(self.particles[:,2])*u[0]*dt + (randn(N)*std[1])
        self.particles[:,2] = self.particles[:,2] + u[1]*dt + (randn(N) * std[2])
        self.particles[:,2] = calc_angle(self.particles[:,2])
        
        
        self.particles[:,3] = self.particles[:,3] + self.particles[:,4]*dt + (randn(N)*std[3])"""
        self.particles[:,0] = self.particles[:,0] + np.cos(self.particles[:,2])*(u[0]+ (randn(N)*std[0]))*dt 
        self.particles[:,1] = self.particles[:,1] + np.sin(self.particles[:,2])*(u[0]+ (randn(N)*std[1]))*dt
        self.particles[:,2] = self.particles[:,2] + dt*(u[1]+randn(N) * std[2])
        self.particles[:,2] = calc_angle(self.particles[:,2])
        
        self.particles[:,3] = self.particles[:,3] + (self.particles[:,4]+ (randn(N)*std[3]))*dt
        self.particles[:,4] = self.particles[:,4] + (randn(N)*std[4]) # mporei na mh thelei
        
    
    def update(self,z,z_phi):
        for i in range(0,self.landmarks.shape[0]) :
            current_lm = self.landmarks[i]
            distance = np.linalg.norm(self.particles[:, 0:2] - current_lm, axis=1)

            temp = calc_angle(np.arctan2(current_lm[1]-self.particles[:,1],current_lm[0]-self.particles[:,0])-self.particles[:,2])
            dist2 = np.array(temp-z_phi[i])
            dist2= calc_angle(dist2)
            full_dist = (np.array([distance,dist2])).T
            #w1 =scipy.stats.norm(distance, self.R_std[0]).pdf(z[i])
            #w2 = scipy.stats.norm(dist2, self.R_std[1]).pdf(z_phi[i]) # gaus or histogram
            
            w1 = [normpdf(np.array([z[i],z_phi[i]]),full_dist[ttt],self.R_std) for ttt in range(0,full_dist.shape[0])]
            self.weights *= w1
            
        self.weights = self.weights + 10**(-10)
        
        self.weights = self.weights / sum(self.weights)
    
    def resample(self): # stratified 
        N = self.N
        positions = (rand(N) + range(N)) / N
    
        indexes = np.zeros(N, 'i')
        cum_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cum_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        
        self.particles[:] = self.particles[indexes]
        self.weights[:] = self.weights[indexes]
        self.weights.fill(1.0 / len(self.weights))
    def multinomial_resample(self):
        cum_sum = np.cumsum(self.weights)
        cum_sum[-1] = 1. 
        indexes= np.searchsorted(cum_sum, rand(len(self.weights)))
        self.particles[:] = self.particles[indexes]
        self.weights[:] = self.weights[indexes]
        self.weights.fill(1.0 / len(self.weights))
    
    
    
    def residual_resample(self):
        N = self.N
        indexes = np.zeros(N, 'i')
    
        
        num_copies = (N*np.asarray(self.weights)).astype(int)
        k = 0
        for i in range(N):
            for _ in range(num_copies[i]): 
                indexes[k] = i
                k += 1
    
        
        residual = self.weights - num_copies     
        residual /= sum(residual)     
        cum_sum = np.cumsum(residual)
        cum_sum[-1] = 1. 
        indexes[k:N] = np.searchsorted(cum_sum, rand(N-k))
        self.particles[:] = self.particles[indexes]
        self.weights[:] = self.weights[indexes]
        self.weights.fill(1.0 / len(self.weights))
    
    def stratified_resample(self):
        N = self.N
        
        positions = (rand(N) + range(N)) / N
    
        indexes = np.zeros(N, 'i')
        cum_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cum_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        
        self.particles[:] = self.particles[indexes]
        self.weights[:] = self.weights[indexes]
        self.weights.fill(1.0 / len(self.weights))
    
    def systematic_resample(self):
        N = self.N
    
        positions = (np.arange(N) + random.random()) / N
    
        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        self.particles[:] = self.particles[indexes]
        self.weights[:] = self.weights[indexes]
        self.weights.fill(1.0 / len(self.weights))
    
    def estimate(self):
        state_vector = self.particles[:,0:5]
        mean = np.average(state_vector,weights = self.weights,axis =0)
        var  = np.average((state_vector - mean)**2, weights=self.weights, axis=0)
        return mean, var


def neff(weights):
    return 1. / np.sum((weights)**2)

if __name__ == "__main__":
    #np.random.seed(223490)
    #np.random.seed(223490)
    control2 = pd.read_csv("dataset/control2.csv", header=None)
    radar2 = pd.read_csv("dataset/radar2.csv", header=None)

    radar2.iloc[:,3] = calc_angle(radar2.iloc[:,3])
    radar2.iloc[:,1] = calc_angle(radar2.iloc[:,1])
    
    initial_landmarks = np.array([[ 3.21846589, 5.35298539], [-3.17810305, 2.97454244]])
    
    alpha = .20
    qstd = np.array([0.01,0.01,0.01,0.001,0.003])
    rstd = np.array([0.5**1,0.3**1])
    dt=0.1
    N=1000
    state_vector = np.array([0,0,0,initial_landmarks[1][0],0.6])
    start = time.time()
    pf = ParticleFilter(N=N,landmarks=initial_landmarks,Q_std = qstd,R_std=rstd,init_x=state_vector)

    x_mean = []
    output_listx=[]
    output_listy=[]
    plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot([], [])
    dif_list=[]
    for i in range(0,100):
        
        u = control2.iloc[i,:]
        z  = np.array([radar2.iloc[i,0],radar2.iloc[i,2]])
        #z[0]= z[0] + randn(1)*rstd[0]
        #z[1]= z[1] + randn(1)*rstd[0]
        z_phi = np.array([radar2.iloc[i,1],radar2.iloc[i,3]])
        #z_phi[0]= z_phi[0] + randn(1)*rstd[1]
        #z_phi[1]= z_phi[1] + randn(1)*rstd[1]
        pf.predict(u,dt)
        pf.update(z,z_phi)
        if neff(pf.weights) < (0.75)*N:
            #print(neff(pf.weights))
            pf.resample()
        
        mu, var = pf.estimate()
        x_mean.append(mu)

        plt.scatter(mu[0], mu[1], marker='o',color='b',zorder=i)
        
        # ------------------enable-disable to add theta---------------------
        plt.quiver(mu[0], mu[1],np.cos(mu[2]), np.sin(mu[2]),linewidths=0.01, edgecolors='k',zorder=i)

        #di= np.sqrt((mu[0]-mu[3])**2+(mu[1]-initial_landmarks[1][1])**2)
        #dif = radar2.iloc[i,2] - di
        #print(dif)
        #dif_list.append(dif)
        ax.scatter(np.asscalar(mu[3]),initial_landmarks[1][1],color='r',zorder=i+1)
        ax.scatter(initial_landmarks[0][0],initial_landmarks[0][1],color='g',zorder=i+1)
        ax.set_title('Particle Filter with moving landmark - iter=%d'%(i+1))
        #plt.scatter(pf.particles[:, 0], pf.particles[:, 1],color='k', marker=',', s=1)
        plt.axis((-6,6,-2,8))
        #sleep(0.07)
        fig.canvas.draw()
        fig.canvas.flush_events()
        #plt.scatter(pf.particles[:, 0], pf.particles[:, 1],color='k', marker=',', s=1,zorder=1)

    x_mean=np.array(x_mean)
    end = time.time()
    print(end - start)
    print(mu)