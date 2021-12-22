# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 09:43:48 2018

@author: wansh
"""
import numpy as np

#Generate the system map
#Can be modified by varied map definition

def genmap(width=500,hight=400,lw=4,lh=2):
    F_map={'width':width,'hight':hight}
    region=[]
    dw=width/lw
    dh=hight/lh
    for h in range(lh):
        for w in range(lw):
            left=(w-1)*dw
            right=left+dw
            down=(h-1)*dh
            up=down+dh
            new_region={}
            new_region['bl']=(1,0,-left,1)
            new_region['br']=(1,0,-right,-1)
            new_region['bd']=(0,1,-down,1)
            new_region['bu']=(0,1,-up,-1)
            region.append(new_region)
    
    region.append(F_map)
    print(region)
    return region

'''
region =
{'bl': (1, 0, 120.0, 1), 'br': (1, 0, -0.0, -1), 'bd': (0, 1, 100.0, 1), 'bu': (0, 1, -0.0, -1)}, 
{'bl': (1, 0, -0.0, 1), 'br': (1, 0, -120.0, -1), 'bd': (0, 1, 100.0, 1), 'bu': (0, 1, -0.0, -1)}, 
{'bl': (1, 0, -120.0, 1), 'br': (1, 0, -240.0, -1), 'bd': (0, 1, 100.0, 1), 'bu': (0, 1, -0.0, -1)}, 
{'bl': (1, 0, -240.0, 1), 'br': (1, 0, -360.0, -1), 'bd': (0, 1, 100.0, 1), 'bu': (0, 1, -0.0, -1)}, 
{'bl': (1, 0, -360.0, 1), 'br': (1, 0, -480.0, -1), 'bd': (0, 1, 100.0, 1), 'bu': (0, 1, -0.0, -1)}, 
{'bl': (1, 0, 120.0, 1), 'br': (1, 0, -0.0, -1), 'bd': (0, 1, -0.0, 1), 'bu': (0, 1, -100.0, -1)}, 
{'bl': (1, 0, -0.0, 1), 'br': (1, 0, -120.0, -1), 'bd': (0, 1, -0.0, 1), 'bu': (0, 1, -100.0, -1)}, 
{'bl': (1, 0, -120.0, 1), 'br': (1, 0, -240.0, -1), 'bd': (0, 1, -0.0, 1), 'bu': (0, 1, -100.0, -1)}, 
{'bl': (1, 0, -240.0, 1), 'br': (1, 0, -360.0, -1), 'bd': (0, 1, -0.0, 1), 'bu': (0, 1, -100.0, -1)}, 
{'bl': (1, 0, -360.0, 1), 'br': (1, 0, -480.0, -1), 'bd': (0, 1, -0.0, 1), 'bu': (0, 1, -100.0, -1)}, 
{'bl': (1, 0, 120.0, 1), 'br': (1, 0, -0.0, -1), 'bd': (0, 1, -100.0, 1), 'bu': (0, 1, -200.0, -1)}, 
{'bl': (1, 0, -0.0, 1), 'br': (1, 0, -120.0, -1), 'bd': (0, 1, -100.0, 1), 'bu': (0, 1, -200.0, -1)}, 
{'bl': (1, 0, -120.0, 1), 'br': (1, 0, -240.0, -1), 'bd': (0, 1, -100.0, 1), 'bu': (0, 1, -200.0, -1)}, 
{'bl': (1, 0, -240.0, 1), 'br': (1, 0, -360.0, -1), 'bd': (0, 1, -100.0, 1), 'bu': (0, 1, -200.0, -1)}, 
{'bl': (1, 0, -360.0, 1), 'br': (1, 0, -480.0, -1), 'bd': (0, 1, -100.0, 1), 'bu': (0, 1, -200.0, -1)}, 
{'bl': (1, 0, 120.0, 1), 'br': (1, 0, -0.0, -1), 'bd': (0, 1, -200.0, 1), 'bu': (0, 1, -300.0, -1)}, 
{'bl': (1, 0, -0.0, 1), 'br': (1, 0, -120.0, -1), 'bd': (0, 1, -200.0, 1), 'bu': (0, 1, -300.0, -1)}, 
{'bl': (1, 0, -120.0, 1), 'br': (1, 0, -240.0, -1), 'bd': (0, 1, -200.0, 1), 'bu': (0, 1, -300.0, -1)}, 
{'bl': (1, 0, -240.0, 1), 'br': (1, 0, -360.0, -1), 'bd': (0, 1, -200.0, 1), 'bu': (0, 1, -300.0, -1)}, 
{'bl': (1, 0, -360.0, 1), 'br': (1, 0, -480.0, -1), 'bd': (0, 1, -200.0, 1), 'bu': (0, 1, -300.0, -1)}, 
{'width': 600, 'hight': 400}]

'''

def j_region(point,region):
    num_r=len(region)-1
    for j_r in range(num_r):
        bound=region[j_r]
        b1=((bound['bl'][0]*point[0]+bound['bl'][1]*point[1]+bound['bl'][2])*bound['bl'][3] >=0)
        b2=((bound['br'][0]*point[0]+bound['br'][1]*point[1]+bound['br'][2])*bound['br'][3] >=0)
        b3=((bound['bd'][0]*point[0]+bound['bd'][1]*point[1]+bound['bd'][2])*bound['bd'][3] >=0)
        b4=((bound['bu'][0]*point[0]+bound['bu'][1]*point[1]+bound['bu'][2])*bound['bu'][3] >=0)
        judg=(b1 and b2 and b3 and b4)
        if judg==1:
            return j_r
    return -1

#Sensor position
def position_sensor(region,num_sensor):
    ep=10.0
    num_r=len(region)-1
    width=region[num_r]['width'] #600
    hight=region[num_r]['hight'] #400
    position_w=np.random.uniform(0.0+ep,float(width)-ep,[num_sensor])
    position_h=np.random.uniform(0.0+ep,float(hight)-ep,[num_sensor])
    p_sensor={'W':position_w,'H':position_h}
    #print(p_sensor)
    return p_sensor

'''
sensor positions =
{'W': [270.12971228, 388.8694851 , 404.3916392 , 442.73727145,
       271.18397591, 274.33007764, 122.29330099, 575.61219842,
       529.95021657,  19.04375833,  26.5504282 , 333.83896925,
       183.17929245, 549.66735177, 266.6356433 , 233.06216865,
       515.03518733, 438.48665717, 110.78767105, 178.53217788], 
       
 'H': [ 25.40442252, 255.96075734, 110.10240911, 287.16616685,
       302.51230276, 210.32429425, 330.35227802, 336.67765204,
       375.68617521, 359.3454687 , 146.70137203, 331.01039873,
       160.8487611 ,  74.21370466, 167.12423782,  49.46813295,
        62.17959722, 160.73615453, 153.02138101, 165.23278849]
}
'''
def gen_datarate(averate,region_rate):
    num=len(region_rate)
    for i in range(num):
        region_rate[i]=np.random.uniform(0.9*averate[i],1.2*averate[i])
    return 1
    
def gen_obs(num_region): # generates locations of obstacles in the map.
    region_obstacle=[[9.61,0.16,1,20],[12.08,0.11,1.6,23],[4.88,0.43,0.1,21]]
    obs=[]
    for i in range(num_region):
        type=np.random.randint(3)
        obs.append(region_obstacle[type])
    print(obs)
    return obs

'''

[
region obstacle =
 [4.88, 0.43, 0.1, 21], [9.61, 0.16, 1, 20], [12.08, 0.11, 1.6, 23], 
 [4.88, 0.43, 0.1, 21], [9.61, 0.16, 1, 20], [9.61, 0.16, 1, 20], 
 [4.88, 0.43, 0.1, 21], [9.61, 0.16, 1, 20], [9.61, 0.16, 1, 20], 
 [4.88, 0.43, 0.1, 21], [12.08, 0.11, 1.6, 23], [12.08, 0.11, 1.6, 23], 
 [9.61, 0.16, 1, 20], [12.08, 0.11, 1.6, 23], [4.88, 0.43, 0.1, 21], 
 [9.61, 0.16, 1, 20], [4.88, 0.43, 0.1, 21], [4.88, 0.43, 0.1, 21], 
 [4.88, 0.43, 0.1, 21], [9.61, 0.16, 1, 20]
]

'''

def gen_gama(g0,d0,the,d):  #input position as array
    gam=np.random.exponential()
#    gama=gam*g0*((d0/np.linalg.norm(P_cen-P_self))**the)
    gama=gam*g0*((d0/d)**the)
    return gama

def list_gama(g0,d0,the,UAVlist,P_cen):
    num_U=len(UAVlist)
    for uk in range(num_U):
        P_self=np.array([UAVlist[uk].position[0],UAVlist[uk].position[1]])
        d=max(50,np.linalg.norm(P_cen-P_self))
        gama=gen_gama(g0,d0,the,d)
        UAVlist[uk].gama=gama
    return 1

def find_pos(position):
    x=np.floor(position[0])
    y=np.floor(position[1])
    return [int(x),int(y)]

# The observations in discrete map nodes
def W_wait(width,height,sensorlist):
    E_wait=np.zeros([height+1,width+1])
    num_sen=len(sensorlist)
    step=1
    for i in range(num_sen):
        pos=find_pos(sensorlist[i].position)
        E_wait[pos[1],pos[0]] +=sensorlist[i].wait
        for j in range(step):
            j+=1
            E_wait[pos[1]+j,pos[0]+j] +=sensorlist[i].wait
            E_wait[max(0,pos[1]-j),max(0,pos[0]-j)] +=sensorlist[i].wait
            E_wait[max(0,pos[1]-j),pos[0]] +=sensorlist[i].wait
            E_wait[pos[1],max(0,pos[0]-j)] +=sensorlist[i].wait
            E_wait[pos[1],pos[0]+j] +=sensorlist[i].wait
            E_wait[pos[1]+j,pos[0]] +=sensorlist[i].wait
    return E_wait
        
        
    
        
        




           
