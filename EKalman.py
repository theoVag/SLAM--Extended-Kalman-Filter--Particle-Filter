# -*- coding: utf-8 -*-
"""
Created on Sat May 25 21:33:06 2019

@author: 
"""

#try:
#    from IPython import get_ipython
#    get_ipython().magic('clear')
#    get_ipython().magic('reset -f')
#except:
#    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from filterpy.stats import plot_covariance_ellipse
from filterpy.common import Q_discrete_white_noise

class ExtendedKalmanFilter:
    
    def __init__(self,x0 = None ,P = None,motion_var = None, measurement_var = None, Q = None, R = None , dt = None):
        
        if(x0 is None):
            self.xt = np.matrix([0,0,0,0,0,0,0]).T
        else:
            self.xt = x0
            
        if(dt is None):
            self.dt=0.1
        else:
            self.dt=dt

        #arxikos pinakas covariance tis gaussian katanomis
        if(P is None):
            self.P = np.matrix([[3,0,0,0,0,0,0],
                                [0,3,0,0,0,0,0],
                                [0,0,np.deg2rad(10),0,0,0,0],
                                [0,0,0,10,0,0,0],
                                [0,0,0,0,10,0,0],
                                [0,0,0,0,0,10,0],
                                [0,0,0,0,0,0,10]])
        else:
            self.P = P
         
        #voithitikos pinakas gia to motion model oste na pigaino apo to 3x3 sto 7x7
        self.Fx = np.matrix([[1,0,0,0,0,0,0],
                             [0,1,0,0,0,0,0],
                             [0,0,1,0,0,0,0]])
        
        if(motion_var is None):
            self.motion_var = 0.01
        else:
            self.motion_var = motion_var
            
        #gia tin kinisi, einai 3x3 kai me ton Fx tha paei sto 7x7
        if(Q is None):
            #self.Q = Q_discrete_white_noise(3, dt=0.1, var=60)
            temp = np.array([[self.motion_var**2,0,0],
                             [0,self.motion_var**2,0],
                             [0,0,np.deg2rad(30)**2]])
            #temp = np.array([[self.motion_var**2,0,0],[0,self.motion_var**2,0],[0,0,self.motion_var**2]])
            self.Q = np.asmatrix(temp)
        else:
            self.Q = Q
            
        if(measurement_var is None):
            self.measurement_var = 0.5
        else:
            self.measurement_var = measurement_var
            
        if(R is None):
            temp = np.identity(2)*(0.5**2)
            temp[1][1] = (0.3**2)
            self.R = np.asmatrix(temp)
        else:
            self.R  = np.identity(2)*(self.measurement_var)
            
        self.Fxj1 = np.matrix([[1,0,0,0,0,0,0],
		                       [0,1,0,0,0,0,0],
		                       [0,0,1,0,0,0,0],
		                       [0,0,0,1,0,0,0],
		                       [0,0,0,0,1,0,0]])

        self.Fxj2 = np.matrix([[1,0,0,0,0,0,0],
		                       [0,1,0,0,0,0,0],
		                       [0,0,1,0,0,0,0],
		                       [0,0,0,0,0,1,0],
		                       [0,0,0,0,0,0,1]])
    
        self.Fxj = [self.Fxj1,self.Fxj2]
            
        self.landmarks_first_seen = 1
        #voithitikes metavlites p.x. gia na kratao ola ta P apo toin arxi os to telos
        self.x = []
        self.PP = []

    def __fxu_at(self, ut):
        
        theta = self.xt.item(2)
        
        v = ut[0]
        w = ut[1]
        
        temp = np.zeros((3,1))
        
        #vlepe sxolio sto predict , giati den vazo ta proigoumena x, y kai theta stin eksisosi
        temp[0][0] = self.dt*v*np.cos(theta)
        temp[1][0] = self.dt*v*np.sin(theta)
        temp[2][0] = self.dt*w#pi2pi(self.dt*w)


        return np.matrix(temp).astype(float)
    
    #jacobian tis fxu_at
    def __F_jacobian_at(self, ut):
        
        temp = np.identity(3)
    
        temp[0][2] = (-self.dt)*ut[0]*np.sin(self.xt[2])
        temp[1][2] = (self.dt)*ut[0]*np.cos(self.xt[2])
    
        temp[0][0] = 0
        temp[1][1] = 0
        temp[2][2] = 0

        return np.matrix(temp).astype(float)
    
    def __Ht_jacobian_at(self,pp):
    
        x = self.xt.item(0)
        y = self.xt.item(1)
        X1 = self.xt.item(3)
        Y1 = self.xt.item(4)
        X2 = self.xt.item(5)
        Y2 = self.xt.item(6)
        if(pp == 1):
            X1 = X2
            Y1 = Y2
        
        q = np.power(x-X1,2) + np.power(y-Y1,2)
        #p = np.power(x-X2,2) + np.power(y-Y2,2) 
        temp = np.zeros((2,5)).astype(float)
        
        temp[0][0] = (-X1+x)/np.sqrt(q)
        temp[0][1] = (-Y1+y)/np.sqrt(q)
        temp[0][3] = (X1-x)/np.sqrt(q)
        temp[0][4] = (Y1-y)/np.sqrt(q)
        
        temp[1][0] = -(-Y1+y)/q
        temp[1][1] = -(X1-x)/q
        temp[1][2] = -1
        temp[1][3] = (-Y1+y)/q
        temp[1][4] = (X1-x)/q
        

    
        return np.matrix(temp).astype(float)
         

    def predict(self, ut):
        #prosoxi i fxu_at , epistrefei peirazei mono ta 3 prota stoixeia tou xt, kai ta ipoloipa i Fx ta midenizei
        #opote an valo mesa stin fxu tin prosthesi me ta proigoumena xt_1, den tha valo gia ti thesi ton landmark 
        #kai tha mou ta midenizei sinexeia. I thesi ton landmark sto predict DEN prepei na allazei katholou
        #an kano 100 fores MONO predict ta X1,Y1,X2,Y2 tha einai idia kai stathera, an sto endiameso kano kai update
        #tote tha allaksoun ta X1,Y1,... apo to update, alla edo tha exoun stathera tin allagmeni timi tous
        self.xt = self.xt + self.Fx.T*self.__fxu_at(ut)
        
        #i kainourgia jacobian vgenei apo tin palia pou einai diastaseon 3x3
        __F_jacobian_at = np.asmatrix(np.identity(7)) + self.Fx.T*self.__F_jacobian_at(ut)*self.Fx

        #ensure angle is between 0,2pi
        self.xt[2] = pi2pi(self.xt[2])
        self.P = __F_jacobian_at*self.P*__F_jacobian_at.T + self.Q#self.Fx.T*self.Q*self.Fx



        return self.xt,self.P

    def update(self, z):
      
        #akomi kai na kano arxikopoiisi sta x,y ton landmark, epeidi trexo to predict
        #ta midenizei, opote vazo edo tin proti ektimisi tis thesis ton landmark
        #vazo ta x kai y, giati exoun allaksei apo to predict kai den einai 0, opos stin arxikopoiisi
        if(self.landmarks_first_seen == 1):
            x = self.xt.item(0)
            y = self.xt.item(1)
            #theta = self.xt.item(2)
            self.xt[3] = x + z.item(0)*np.cos(z.item(1))
            self.xt[4] = y + z.item(0)*np.sin(z.item(1))
            self.xt[5] = x + z.item(2)*np.cos(z.item(3))
            self.xt[6] = y + z.item(2)*np.sin(z.item(3))
            self.landmarks_first_seen = 0
            #dokimi me cos(z.item() + theta deb doulepse)         
          
        #pragmatiki metrisi
        z_real = [z[0:2], z[2:4]]
        
        #gia kathe empodio
        for t in range(2): 
            x = self.xt.item(0)
            y = self.xt.item(1)
            theta = pi2pi(self.xt.item(2)) 
            X1 = self.xt.item(3)
            Y1 = self.xt.item(4)

            if(t==1):
                X1 = self.xt.item(5) # diladi = X2
                Y1 = self.xt.item(6) # diladi = Y2

            
            q = np.power(x-X1,2) + np.power(y-Y1,2)
           
            zt_est = np.matrix([[np.sqrt(q)],[pi2pi(np.arctan2(Y1-y,X1-x)-theta)]])
        
   
            Hti = self.__Ht_jacobian_at(t)*self.Fxj[t]
            
            K = self.P*Hti.T*(Hti*self.P*Hti.T + self.R).I 

            tmp = (z_real[t]-zt_est)
            tmp[1]=pi2pi(tmp[1])
            #tmp = self.__residual(z_real[t],zt_est)
            #tmp[1] = tmp[1]%(2*np.pi)- np.pi
            #print(tmp)
            self.xt = self.xt + K*(tmp)
            #self.xt[2] = pi2pi(self.xt[2])
            self.P = (np.matrix(np.identity(7)) - K*Hti)*self.P
            

        self.x.append(self.xt)
        self.PP.append(self.P)
        #self.xt = self.xt
        #self.P = self.P

        return self.xt,self.P
    
def pi2pi(a):
#    if(a<0):
#        a = a % (2 * np.pi) * (-1)
#    else:
    a = a % (2 * np.pi)
    
    if a > np.pi:             # move to [-pi, pi)
        a -= 2 * np.pi
        
    return a

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]       

def plotEllipse(axis,x,y,P):
    #https://stackoverflow.com/questions/20126061/creating-a-confidence-ellipses-in-a-sccatterplot-using-matplotlib
    #http://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
    nstd = 0.8
    
    cov = P

    vals, vecs = eigsorted(cov)
    vals = vals[0:2]
    vecs = vecs[0:2]
    
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 2 * nstd * np.sqrt(vals)
    ell = Ellipse(xy=(x, y),
              width=w, height=h,
              angle=theta, color='black')
    ell.set_facecolor('none')
    axis.add_artist(ell)
    #plt.scatter(x, y)
    plt.show()
    
    
    
    
    
    
def example():

    #dt=0.1
    
    df1 = pd.read_csv("dataset/control1.csv", header=None )
    df1 = np.array(df1)
    df2 = pd.read_csv("dataset/radar1.csv", header=None )
    df2 = np.array(df2)
    
#    for kk in range(df1.shape[0]):
#        df1[kk][1] = df1[kk][1] % (2*np.pi) 
    #print(df1[0])
    #exo euros 0,2pi alla kai me arnitika mesa kai ta kano ola thetika
    for kk in range(df2.shape[0]):
        df2[kk][1] = pi2pi(df2[kk][1] )
        df2[kk][3] = pi2pi(df2[kk][3] )


    #Q = np.matrix([[2.5e-06,0,0],[0,1.0e-03,0],[0,0,1.0e-01]])
    Q = np.matrix([[0.01**2,0,0,0,0,0,0],
                   [0,0.01**2,0,0,0,0,0],
                   [0,0,0.01**2,0,0,0,0],
                   [0,0,0,0.001,0,0,0],
                   [0,0,0,0,0.001,0,0],
                   [0,0,0,0,0,0.001,0],
                   [0,0,0,0,0,0,0.001]])

    #m0 = np.matrix([0,0,0,1,2,3,4]).T
    #ek = ExtendedKalmanFilter(motion_var = 2 ,measurement_var = 0.5, dt = 0.1)
    ek = ExtendedKalmanFilter( Q=Q,measurement_var = 0.5, dt = 0.1)

    #old plot
    fig = plt.figure()
    axRobotPosition = plt.subplot(111) 
    axRobotPosition.set_xlim(-10, 10)
    axRobotPosition.set_ylim(-5, 10)
    
    #axRobotPosition.set_xlim(-3, 15)
    #axRobotPosition.set_ylim(-8, 8)
    temp_x = []
    temp_y = []
    for i in range(df1.shape[0]):#df1.shape[0]
    
        ek.predict(list(df1[i]))
        #print(ek.xt.item(2))
        zt = np.matrix([[df2[i][0]],[df2[i][1]],[df2[i][2]],[df2[i][3]]])
        ek.update(zt)
        temp_x.append(ek.xt.item(0))
        temp_y.append(ek.xt.item(1))
        #print(ek.xt.item(2))
        lambda_, v = np.linalg.eig(ek.P[0:2, 0:2])
        lambda_ = np.sqrt(lambda_)
        print(lambda_)
        #old plot
        #axRobotPosition.clear()
    #    axRobotPosition.plot(ek.xt.item(0), ek.xt.item(1), marker=(3, 0, math.degrees(ek.xt.item(2))),color='green', markersize=20, linestyle='None')
    #    axRobotPosition.plot(ek.xt.item(0), ek.xt.item(1), 'bo')
    #    axRobotPosition.quiver(ek.xt.item(0), ek.xt.item(1),np.cos(ek.xt.item(2)), np.sin(ek.xt.item(2)),linewidths=0.01, edgecolors='k')
    #    axRobotPosition.quiver(ek.xt.item(0), ek.xt.item(1),ek.xt.item(0), ek.xt.item(1),linewidths=0.01, edgecolors='k')
    #    axRobotPosition.scatter(ek.xt.item(0), ek.xt.item(1), s=20, c = 'b')
    #    axRobotPosition.plot(temp_x,temp_y,color='blue')
        #plotEllipse(axRobotPosition,ek.xt.item(0), ek.xt.item(1),ek.P[0:2, 0:2])#if (i%5 == 0 ):    
        plot_covariance_ellipse((ek.xt[0,0], ek.xt[1,0]), ek.P[0:2, 0:2],std=1, facecolor='lightskyblue', alpha=0.05)
        plot_covariance_ellipse((ek.xt[3,0], ek.xt[4,0]), ek.P[3:5, 3:5],std=1, facecolor='lightcoral', alpha=0.05)
        plot_covariance_ellipse((ek.xt[5,0], ek.xt[6,0]), ek.P[5:7, 5:7],std=1, facecolor='lightgreen', alpha=0.05)
        axRobotPosition.scatter(ek.xt.item(0), ek.xt.item(1), s=20, c = 'b')
        axRobotPosition.quiver(ek.xt.item(0), ek.xt.item(1),np.cos(ek.xt.item(2)), np.sin(ek.xt.item(2)),linewidths=0.01, edgecolors='k')
        axRobotPosition.plot(ek.xt.item(3), ek.xt.item(4), 'rx')
        #plotEllipse(axRobotPosition,ek.xt.item(3), ek.xt.item(4),ek.P)
        axRobotPosition.plot(ek.xt.item(5), ek.xt.item(6), 'gx')
        #axRobotPosition.legend(('robot', 'landmark1', 'landmark2'))
        axRobotPosition.set_title('Kalman filter with static landmarks')
        axRobotPosition.relim()
        axRobotPosition.autoscale_view()
        #axRobotPosition.set_aspect('equal')
            
        fig.canvas.draw()
        fig.canvas.flush_events()
        #plt.pause(1)


    #print(type(ek.xt))    
    #print(ek.Q)
    #print(ek.xt)
    #print(ek.z_es)
    #print(len(xt[0]))
    print(ek.xt)
    #print(ek.P)

    #ani.save('sine_wave_2.gif', writer='pillow')   
    return ek.xt,ek.P,ek.x,ek.PP#ek.z_real, ek.z_es, ek.res


    
if __name__ == '__main__':
    xt,P,x,PP = example()
    
    
    
    
    

