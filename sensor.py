# -*- coding: utf-8 -*-

from gmap import j_region
import numpy as np
import random

class sensor_agent:
    def __init__(self, position, C, region, data_rate, slot):
        self.position = position.copy()
        self.capacity = C                           #2000
        self.rNo = j_region(self.position, region)  # it will be either -1 else it will be the region
        self.databuf = 0                            # data buffer of the sensor
        self.data_rate = data_rate[self.rNo]        # give a data rate to each sensor based on the region
        self.slot = slot                            #0.5
        self.wait = 0
        
    
    def fresh_buf(self,UAVlist):  #accumulate data in the former slot, transmit to UAV
        distance=[]
        num=len(UAVlist)          # 6
        self.databuf=self.databuf+np.random.poisson(self.data_rate*self.slot) #add some data in sensor buffer
#        print(self.databuf)
        for uk in range(num):
            p1=np.array([UAVlist[uk].position[0],UAVlist[uk].position[1]]) #take position of UAV
            p2=np.array([self.position[0],self.position[1]])  #take position of sensor
            distance.append(np.linalg.norm(p1-p2))            # calculte distance between them
        
        min_d=min(distance)  #find minimum among all distances so that we know which UAV is closest
        temp=[]              # will store sorted distances
        inf=1e15

        for uk in range(num):
            md=min(distance)          #find the min distance
            if md>min_d+1:
                break
            l0=distance.index(md)     # find index of the min distance
            temp.append(l0)           # add min distance index in temp
            distance[l0]=inf          # change the min distanace value to infinity

        min_idx=random.sample(temp,1)[0]

        if(min_d>UAVlist[min_idx].r):         # if minimum distance is greater than the communication radius
            self.wait=self.wait+1
            return -1
        else:                             #else part need to understand
            UAVlist[min_idx].data_buf=UAVlist[min_idx].data_buf+min(self.databuf,self.slot*self.capacity)
            pre_buf=self.databuf
            self.databuf=max(0,self.databuf-self.slot*self.capacity)
            self.wait=self.databuf*self.wait/pre_buf   #eq 18 from algorithm 2
            return min_idx


