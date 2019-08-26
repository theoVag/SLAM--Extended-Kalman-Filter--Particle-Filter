# SLAM--Extended-Kalman-Filter--Particle-Filter
Localization -SLAM problem with Extended Kalman filter and Particle filter

This repository consist of three source codes dealing with three different occasions.
In the first one, a vehicle is moving in the 2d space. The measurement model includes the distance and the angle between the vehicle and the 2 obstacles. The goal is to estimate as good as possible the pose of the vehicle ([x,y,theta]) and the position of the 2 obstacles (7 states [x,y,theta,xob1,yob1,xob2,yob2]). The problem assumes gaussian noise with zero mean and std of 1.
The measurements contain gaussian noise with zero mean value.

In the second one, obstacles are consider to be at the positions calculated in the previous stage. This time a particle filter is used to estimate the vehicle's pose.

And at the last one, the 1st obstacle is considered at the calculated position but the second obstacle has the previous calculated position as initial and it is moving along the x-axis with a constant speed value. The goal is to estimate the vehicle's pose and the x-position and speed of the 2nd obstacle (state vector [x,y,theta,xob2,vob]).

The dataset includes control values (linear speed,rotational speed) and the measurement values (distance1,angle1,distance2,angle2).


Motion and measurements models are presented in the full report (main language is greek). There is also an estimation procedure for proposing values about uncertainty, noises and other used parameters. Parameters for the filters were estimated after differents test.

Notes:
The repository contains 2 implementations of the Extended Kalman Filter- one simple approach and one exploiting sparse arrays (useful for large problem with multiple obstacles) based on the book Probabilistic Robotics (http://www.probabilistic-robotics.org/) and adapted to the current problem.

Particle filter contains different type of resampling methods that can be used by changing the name of the function used.
It contains also two different types of particles initialization - Uniform and Gaussian.
In order to implement update step and exploit both distance and angle measurements, a multivariate normal distribution was used.
