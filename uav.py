# -*- coding: utf-8 -*-

from gmap import j_region
from gmap import W_wait,find_pos
import numpy as np

class UAV_agent:
    def __init__(self,uk,com_r,region_obstacle,region,omeg,slot,t_bandwidth,cal_L,k,f_max,p_max):
        self.position = [np.random.uniform(0.0, region[-1]['width']), np.random.uniform(0.0, region[-1]['hight'])]
        self.No=uk
        self.r = com_r
        self.region_ifo=region
        self.region_No=j_region(self.position,region)
        self.obs=region_obstacle[self.region_No]
        self.data_buf=0
        self.hight=0   #modified by cal_hight
        self.v=[0,0]
        self.p_tr=p_max    #modified by cal_ptr
        self.f=0       #modified by cal_f
        self.bandwidth=t_bandwidth
        self.slot=slot
        self.L=cal_L
        self.alpha=0.1
        self.gama=0.1
        self.noise=1
        self.D_l=0    #modified by fresh_buf
        self.D_tr=0   #modified by fresh_buf
        self.omeg=omeg
        self.k=k
        self.f_max=f_max
        self.formerstate=np.zeros((84,84,2))
        self.nowstate=np.zeros((84,84,2))
        self.reward=0
    
    def resetposition(self): #Set the initial UAV position in random mode
        self.position=[np.random.uniform(0.0,self.region_ifo[-1]['width']),np.random.uniform(0.0,self.region_ifo[-1]['hight'])]
    
    def cal_hight(self,b_up=1e4,ite=15):   #calculate the optimal UAV height
        a=self.obs[0]
        b=self.obs[1]
        eta0=self.obs[2]
        eta1=self.obs[3]
        L=0   #<0
        H=b_up  #>0
        for i in range(ite):
            mid=(L+H)/2.0
            ep=np.exp( -b*(np.arctan(mid)-a) )
            div=2*mid*(eta1+(eta0-eta1)/(1+a*ep))+(eta0-eta1)*a*b*ep/((1+a*ep)**2)
            if div<0:
                L=mid
            elif div>0:
                H=mid
            else:
                self.hight=self.r*mid
                return self.hight
        
        self.hight=self.r*mid
        return self.hight
    
    def fresh_buf(self):   # update the data queue using Eq 8 and Eq 9
        self.D_l = self.slot*self.f/self.L
        self.D_tr = self.alpha*self.omeg*self.bandwidth*self.slot*np.log2(1+self.gama*self.p_tr/(self.alpha*self.noise* self.bandwidth))
        self.data_buf = max(self.data_buf-self.D_l-self.D_tr, 0)
        return self.data_buf
        
    def cal_f(self,V):      #calculate local frequency
        cf=np.sqrt( (self.slot*self.data_buf)/(3*self.L*V*self.omeg*self.k) )
        if cf<=self.f_max:
            self.f=cf
            return cf
        else:
            self.f=self.f_max
            return self.f_max
        
    def cal_ptr(self,p_max,V,noise):        #calculate transmition power to the cloud center
        self.noise=noise
        ptr=self.bandwidth*self.alpha*(self.data_buf*self.slot)/(V*self.omeg*np.log(2))-self.bandwidth*self.alpha*noise/self.gama 
        self.p_tr=min(p_max,max(ptr,0))
        return self.p_tr
    
    def cal_alpha(self,dlambda,alfmin,ite,alpha_U):
        h=self.data_buf*self.bandwidth*self.slot    #different only because of data_buf
        M=self.gama*self.p_tr/(self.noise*self.bandwidth)
        h=h
        M=M
        alpha_L=alfmin
#        v_L=-h*np.log2(1+M/alpha_L)+h*M/(np.log(2)*(alpha_L+M))+dlambda
#        v_U=-h*np.log2(1+M/alpha_U)+h*M/(np.log(2)*(alpha_U+M))+dlambda
#        if v_L>0:
#            self.alpha=alfmin
#            return alfmin
#        elif v_U<0:
#            self.alpha=alpha_U
#            return alpha_U
        for i in range(ite):
            mid=(alpha_L+alpha_U)/2
            v_mid=-np.log2(1+M/mid)+M/(np.log(2)*(mid+M))+dlambda/(h+1e-3)
            if v_mid<0:
                alpha_L=mid
            elif v_mid>0:
                alpha_U=mid
            else:
                self.alpha=mid
                return mid
        self.alpha=mid
        return mid


#update the UAV position according to the path    
    def fresh_position(self, v, region_obstacle):   #out=1 represents moving out of map
        out = 0
        self.v = []
        self.v = v.copy() 
        width = self.region_ifo[-1]['width']
        hight = self.region_ifo[-1]['hight']
        self.position[0] = self.position[0]+v[0]
        self.position[1] = self.position[1]+v[1]
#        l0=self.r/np.sqrt(2)
        l0 = 0
        # x cooridnate
        if self.position[0] >= width-l0:
            self.position[0] = width-l0
            out = 1
        elif self.position[0] <= l0:
            out = 1
            self.position[0] = l0
        
        # y coordinate 
        if self.position[1] >= hight-l0:
            self.position[1] = hight-l0
            out = 1
        elif self.position[1] <= l0:
            out = 1
            self.position[1] = l0

        self.region_No = j_region(self.position, self.region_ifo)
        self.obs = region_obstacle[self.region_No]
        return out

            

# Generate the local observation (Can be varied by different observation definition)

    def map_feature(self, datarate, UAVlist, E_wait):     # return 84 84 2 feature
        size_f = 84                                       # 84  Oij RxR
        size_h = size_f/2                                 # 42  
        sight = 3                                         # 3
        position = np.zeros([size_f, size_f, 2])          # 84x84x2 array  Pkij 
        feature = np.zeros([size_f, size_f, 1])           # 84x84x1 array  O 
        inrange = []
        num_uav = len(UAVlist)                            # 6

        for uk in range(num_uav):                         # find neighbor UAVs
            ps = UAVlist[uk].position                     # Take the UAV position
            No = UAVlist[uk].No                           # Take the UAV number
            if No == self.No:
                inrange.append(No)
                continue
            
            # x cordinate positions
            condition1 = (ps[0] >= self.position[0] - (size_h-1) * sight - self.r) 
            condition2 = (ps[0] <= self.position[0] + size_h * sight + self.r)

            # y coordinate position
            condition3 = (ps[1] >= self.position[1] - (size_h-1) * sight - self.r)
            condition4 = (ps[1] <= self.position[1] + size_h * sight + self.r)

            if condition1 and condition2 and condition3 and condition4:
                inrange.append(No)
        
        # position array
        for f in range(size_f):                          # define positions of each points in the feature
            position[:, f, 0] = self.position[0] - (size_h-1)*sight + sight*f  # x - 123 + 3*{0,1,2...83}
            position[f, :, 1] = self.position[1] + (size_h-1)*sight - sight*f  # y + 123 - 3*{0,1,2,...83}
        
        # feature array 
        for i in range(84):
            for j in range(84):

                c1 = position[i, j, 0] < 0
                c2 = position[i, j, 0] > self.region_ifo[-1]['width']

                c3 = position[i, j, 1] < 0
                c4 = position[i, j, 1] > self.region_ifo[-1]['hight']
                
                # if any condition is true then skip the inner loop
                if c1 or c2 or c3 or c4:
                    continue
                
                #  if it is in the region then only update feature i.e. observations
                r_no = j_region([position[i, j, 0], position[i, j, 1]], self.region_ifo)                 # the region No with the current point in
                pos = find_pos(position[i, j, :])                                                        # returns the floor value
                feature[i, j, 0] = datarate[r_no] * E_wait[pos[1], pos[0]]  #  20  * 301 * 401

                for k in range(len(inrange)):

                    d = np.linalg.norm(np.array([position[i, j, 0]-UAVlist[inrange[k]].position[0], position[i, j, 1]- UAVlist[inrange[k]].position[1]]))
                    # print(" for", k , "th UAV", "distance is:",d, "self. UAV radius is",self.r, "\n" )
                    if d <= self.r:                        # if distance is less than the radius
                        if inrange[k] == self.No:
                            continue
                        else:
                            if feature[i, j, 0] > 0:      # if feature > 0 then make them 0 else let them whatever they are
                                feature[i, j, 0] = 0

                            feature[i, j, 0]=feature[i, j, 0] - 8000 # subtract 8000 from the features #eq 19
                   # print(" for", k , "th UAV", "feature is",  "\n" )
        feature = feature/100    # we will get negative feature values
#                            break
        return feature.copy()     # Ok(tp)
