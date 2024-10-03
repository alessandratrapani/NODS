import numpy as np
import math as m
import pandas as pd
from scipy import spatial
import dill
import os
import time

class NODS:
    def __init__(self,model_parameters):
        self.tauCa   = model_parameters['production']['tauCa']
        self.tauNOS1 = model_parameters['production']['tauNOS1']
        self.tauNOS2 = model_parameters['production']['tauNOS2']
        self.A       = model_parameters['production']['A']
        self.B       = model_parameters['diffusion']['B']
        self.D       = model_parameters['diffusion']['D'] # diffusion coefficient [um^2/ms]
        self.I       = model_parameters['diffusion']['I'] # inactivation coefficient [1/ms] 

        self.ds      = model_parameters['diffusion']['ds']
        self.r_max   = model_parameters['diffusion']['r_max']
        self.distances = np.arange(-self.r_max, self.r_max+self.ds, self.ds)
        r_2    = self.distances**2
        self.Green_LUT = np.zeros((len(self.distances),2))
        self.Green_LUT[:, 0] = Green_function(0, r_2, self.D, self.I)
        self.Green_LUT[:, 1] = Green_function(2, r_2, self.D, self.I)

        #self.dt         = model_parameters['simulation']['dt']
        self.dt = 2
        self.time       = np.arange(model_parameters['simulation']['t_start'], model_parameters['simulation']['t_end'], self.dt)
        self.Calm2C_0   = model_parameters['simulation']['Calm2C_0']
        self.nNOS_0     = model_parameters['simulation']['nNOS_0']
        self.NO_p_0     = model_parameters['simulation']['NO_p_0']

    def init_geometry(self, nNOS_coordinates, ev_point_coordinates, source_ids, nos_ids = None, cluster_nos_ids=None, ev_point_ids = None, cluster_ev_point_ids=None, file_ev_points = None, file_nNOS = None):

        if file_ev_points is None:
            if ev_point_ids is None:
                ev_point_ids = np.arange(len(ev_point_coordinates))
            if cluster_ev_point_ids is None:
                cluster_ev_point_ids = -1*np.ones((len(ev_point_coordinates)))

            ev_points = {}    
            for index,ev_point_id in enumerate(ev_point_ids):
                ev_points[ev_point_id] = dict(x =  ev_point_coordinates[index,0],
                                                   y =  ev_point_coordinates[index,1],
                                                   z =  ev_point_coordinates[index,2],
                                                   cluster = cluster_ev_point_ids[index])
            #ev_points.sort_values(by='evpoint_id')
        #TODO else: load da file

        if file_nNOS is None:
            if nos_ids is None:
                nos_ids = np.arange(len(nNOS_coordinates))
            if cluster_nos_ids is None:
                self.cluster_nos_ids = -1*np.ones((len(nNOS_coordinates)))
            else:
                self.cluster_nos_ids = cluster_nos_ids

            all_nNOS = {}
            for index,nos_id in enumerate(nos_ids):
                all_nNOS[nos_id] = dict(source_id = source_ids[index],
                                            x =  nNOS_coordinates[index,0],
                                            y =  nNOS_coordinates[index,1],
                                            z =  nNOS_coordinates[index,2],
                                            cluster = self.cluster_nos_ids[index])                    
            #all_nNOS = pd.DataFrame({'source_id':source_ids, 'nos_id':nos_ids, 'x': nNOS_coordinates[:,0], 'y': nNOS_coordinates[:,1], 'z': nNOS_coordinates[:,2]})
        #TODO else: load da file
        self.sort_sources(all_nNOS, ev_points)
        self.no_conc = np.zeros(len(ev_point_ids))
        #df = pd.DataFrame(ev_points)
        #df.to_csv("/home/nomodel/code/NODS/results/NO_concentration_data/ev_points_dict.csv")
        return

    def sort_sources(self, all_nNOS, ev_points, filename = None):
        """function to filter the sources of nNOS activation to be avaluated"""
        self.relative_dist = []
        self.source_to_eval = []
        cluster_ids = np.unique(self.cluster_nos_ids)
        # loop on cluster DA PARALLELIZZARE
        
        for cluster in [cluster_ids[0]]:
            # loop on the receiver
            for evpoint_id in ev_points:
                # loop on the sources
                if ev_points[evpoint_id]['cluster']==cluster:
                    ev_point_coordinates = np.array([ev_points[evpoint_id]['x'],ev_points[evpoint_id]['y'],ev_points[evpoint_id]['z']])
                    for nos_id in all_nNOS: 
                        if all_nNOS[nos_id]['cluster']==cluster:
                            nNOS_coordinates = np.array([all_nNOS[nos_id]['x'],all_nNOS[nos_id]['y'],all_nNOS[nos_id]['z']])                    
                            # distance evaluation
                            d = spatial.distance.euclidean(nNOS_coordinates, ev_point_coordinates)
                            # check on relevant distance value
                            if d < self.r_max:
                                # lists update
                                source_id = all_nNOS[nos_id]['source_id']
                                self.source_to_eval.append(source_id)
                                self.relative_dist.append([int(source_id), int(nos_id), int(evpoint_id), d, int(cluster)]) # 0: id_source, 1: id_nos, 2:id_evpoint, 3: relative_distance
        # elimination repetition of same source
        self.source_to_eval = np.unique(self.source_to_eval)
        df_relative_dist = pd.DataFrame(self.relative_dist)
        df_relative_dist.to_csv('relative_dist.csv',header=False)

        return
    
    def init_simulation(self,simulation_file, number_of_evaluation_points, store_sim=True):

        self.NO_from_source = {}
        for source_id in self.source_to_eval:
            self.NO_from_source[source_id] = dict(  Calm2C = self.Calm2C_0 ,
                                                    nNOS   = self.nNOS_0,
                                                    NO_produced_t0  = self.NO_p_0,
                                                    u               = np.zeros_like(self.distances).astype(np.float64),
                                                    NO_diffused_tf  = 0
                                                )
            
        self.NO_in_ev_points = np.zeros((number_of_evaluation_points))
        
        if store_sim:
            dill.dump(self, open(simulation_file, "wb"))
        return 
    
    def load_simulation(self,simulation_file):
        return dill.load(open(simulation_file, "rb"))
    
    def store_simulation(self,simulation_file):
        dill.dump(self, open(simulation_file, "wb"))
        return 
        
    def evaluate_diffusion(self,active_sources,t):

        source_data = self.NO_from_source
        source_to_eval = self.source_to_eval
        dt = self.dt
        tauCa = self.tauCa
        tauNOS1 = self.tauNOS1
        tauNOS2 = self.tauNOS2
        A = self.A
        B = self.B
        Green_LUT = self.Green_LUT
        r_max = self.r_max
        ds = self.ds
        NO_in_ev_points = self.NO_in_ev_points
        #no_conc_to_file = self.no_conc
        no_conc_to_file = []
        output_folder = "/home/nomodel/code/NODS/results/NO_concentration_data_2ms/"
        if not os.path.exists(output_folder):
                os.makedirs(output_folder)

        file_name = f"NO_concentration_t_{t}.csv"
        file_path = os.path.join(output_folder, file_name)
    
        for source_id in source_to_eval:
            spike = 1 if source_id in active_sources else 0

            if (source_id == 3303) & (3303 in active_sources):
                print(f'spike 3303, {t}')

            if source_id == 3303:
                time.sleep(0.01) 

            source = source_data[source_id]
            nNOS, Calm2C, NO_produced_t1 = Production_function(dt, spike, source['Calm2C'], source['nNOS'], tauCa, tauNOS1, tauNOS2, A)
            u, NO = Diffusion_function(dt, source['u'], Green_LUT, source['NO_produced_t0'], NO_produced_t1, B)

            source['Calm2C'] = Calm2C
            source['nNOS'] = nNOS
            source['NO_produced_t0'] = NO_produced_t1
            source['u'] = u
            source['NO_diffused_tf'] = NO 
        


            no_conc_to_file.append([NO_produced_t1,source_id,nNOS,Calm2C])
        df_no_conc = pd.DataFrame(no_conc_to_file)
        df_no_conc.to_csv(file_path, header=False)

        for row in self.relative_dist:
            source_id, nos_id, ev_points_id, d, cluster = row
            if d < 0.2:
                d = 0.2

            distance_index = round((d + r_max) / ds)
            NO_contribution = source_data[source_id]['NO_diffused_tf'][distance_index]
            NO_in_ev_points[ev_points_id] += NO_contribution
            """
            no_conc_to_file[ev_points_id] = NO_in_ev_points[ev_points_id]
        df_no_conc = pd.DataFrame(no_conc_to_file)
        df_no_conc.to_csv(file_path,header=False)"""  
                
        return

def Production_function(dt,Ca_spike,Calm2C_old,nNOS_old,tauCa,tauNOS1,tauNOS2,A):
    
    Calm2C = Calm2C_old + (((Calm2C_old/tauCa) + Ca_spike)*dt)
    a = ((1/tauNOS1)*((Calm2C)/((Calm2C)+1)))-(nNOS_old/tauNOS2)
    nNOS = nNOS_old+a*dt
    NO = nNOS*A

    return nNOS, Calm2C, NO

def Green_function(t,r_2,D,I):

    eps = 0.1
    if t == 0:
        t = eps

    a = 1 / (4 * m.pi * D * t)
    e1 = (-1 * r_2) / (4 * D * t)
    exp_diffusion = np.exp(e1)
    exp_inactivation = np.exp(-I * t)
    G = (m.pow(a, 3 / 2)) * exp_diffusion * exp_inactivation    

    return G

def Diffusion_function(dt,u0,Green_LUT,NO_produced_t0,NO_produced_t1, B):

    spacial_conv = np.convolve(Green_LUT[:,1], u0, 'same')    
    u = spacial_conv + (((Green_LUT[:,0]*NO_produced_t1) + (Green_LUT[:,1]*NO_produced_t0))*((dt)/2))    
    NO = u*B 

    return u, NO