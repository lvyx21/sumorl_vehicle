import os
import sys
from typing import Callable, List, Union

from gymnasium.spaces import Box

import os
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple, Union


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")
import gymnasium as gym
import numpy as np
import pandas as pd
import sumolib
import traci
from gymnasium.utils import EzPickle, seeding
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

from .observations import DefaultObservationFunction, ObservationFunction
from .traffic_signal import TrafficSignal



LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
from gymnasium import spaces
class VehicleController:
    MIN_GAP = 2.5

    def __init__(self, sumo, env,vehicle_id):
        self.vehicle_id=vehicle_id
        self.sumo = sumo
        self.env = env
        self.id = vehicle_id
        self.current_speed = self.sumo.vehicle.getSpeed(self.id)
        self.next_action_time = env.sim_step
        self.delta_time=1
        self.num_surrounding_vehicles=5
        #num_traffic_signals=len(self.env.ts_ids)
        #num_observations=4+len(self.env.ts_ids)+self.num_surrounding_vehicles*4+len(self.env.ts_ids)*2
        num_observations=5+len(self.env.ts_ids)
        self.observation_space=spaces.Box(low=0,high=1,shape=(num_observations,),dtype=np.float32)
        self.action_space = Box(low=-self.sumo.vehicle.getDecel(self.id), high=self.sumo.vehicle.getAccel(self.id), shape=(1,), dtype=np.float32)
        self.next_action=None
        

    def update(self):
        self.current_speed=self.sumo.vehicle.getSpeed(self.id)
        if self.vehicle_time_to_act():
            self.excute_action() 
    
    def execute_action(self):
        if self.next_action is not None:
            self.set_next_action(self.next_action)
            self.next_action=None

    def set_next_action(self, action):
        """
        Sets the next action for the vehicle which could be accelerate, maintain, or decelerate.

        Args:
            action (str): One of ['accelerate', 'maintain', 'decelerate']
        """
        if self.env.sim_step < self.next_action_time:
            return  # 未到下一次行为设置时间
        new_speed = self.current_speed + action * self.delta_time
        self.sumo.vehicle.setSpeed(self.id, new_speed)
        self.next_action_time = self.env.sim_step + self.delta_time
        

    def vehicle_time_to_act(self):
        return self.next_action_time <= self.env.sim_step

    
    
    def get_observation(self):
        max_speed = self.sumo.vehicle.getMaxSpeed(self.id)
        max_length = self.sumo.lane.getLength(self.sumo.vehicle.getLaneID(self.id))
        
        # 获取当前车辆的速度
        this_speed = self.sumo.vehicle.getSpeed(self.id)
        
        # 获取领先车辆的信息
        lead_id = self.sumo.vehicle.getLeader(self.id)
        if lead_id in ["", None]:
            lead_speed = max_speed
            lead_head = max_length
        else:
            lead_speed = self.sumo.vehicle.getSpeed(lead_id)
            lead_head = self.sumo.vehicle.getPosition(lead_id)[0] - self.sumo.vehicle.getPosition(self.id)[0] - self.sumo.vehicle.getLength(self.id)
        
        # 获取跟随车辆的信息
        follower_id = self.sumo.vehicle.getFollower(self.id)
        if follower_id in ["", None]:
            follow_speed = 0
            follow_head = max_length
        else:
            follow_speed = self.sumo.vehicle.getSpeed(follower_id)
            follow_head = self.sumo.vehicle.getHeadway(follower_id)
        
        # 归一化并组合观测
        observation = [
            this_speed / max_speed,
            (lead_speed - this_speed) / max_speed,
            lead_head / max_length,
            (this_speed - follow_speed) / max_speed,
            follow_head / max_length
        ]

        for ts_id in self.env.ts_ids:
            traffic_signal_state = self.sumo.trafficlight.getRedYellowGreenState(ts_id)
            observation.append(traffic_signal_state)

        return np.array(observation, dtype=np.float32)
        
    
        
    def compute_reward(self):
        """
        Compute the reward for the vehicle.
        """
        # 示例奖励函数：根据车辆速度给予奖励，速度越高奖励越高
        speed = self.sumo.vehicle.getSpeed(self.id)
        waiting_time=self.sumo.vehicle.getWaitingTime(self.id)
        time_loss = self.sumo.vehicle.getTimeLoss(self.id) 
        type2_waiting_time = sum(self.sumo.vehicle.getWaitingTime(veh) for veh in self.sumo.vehicle.getIDList() if self.sumo.vehicle.getTypeID(veh) == 'type2')
        type2_time_loss = sum(self.sumo.vehicle.getTimeLoss(veh) for veh in self.sumo.vehicle.getIDList() if self.sumo.vehicle.getTypeID(veh) == 'type2')
   
    
        reward =-time_loss+speed/self.sumo.vehicle.getMaxSpeed(self.id)-waiting_time
        return reward

    