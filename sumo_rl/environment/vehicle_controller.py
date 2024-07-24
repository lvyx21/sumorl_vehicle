import os
import sys
from typing import Callable, List, Union



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
        num_traffic_signals=len(self.env.ts_ids)
        num_observations=4+len(self.env.ts_ids)+self.num_surrounding_vehicles*4+len(self.env.ts_ids)*2
        self.observation_space=spaces.Box(low=0,high=1,shape=(num_observations,),dtype=np.float32)
        self.action_space = spaces.Discrete(3)
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
        if action == 1:
            self.accelerate()
        elif action == 0:
            self.maintain()
        elif action == -1:
            self.decelerate()
        else:
            raise ValueError(f"Unknown action: {action}")
        self.next_action_time = self.env.sim_step + self.delta_time       
        

    def vehicle_time_to_act(self):
        return self.next_action_time <= self.env.sim_step

    def accelerate(self):
        """
        Increase the vehicle's speed.
        """
        new_speed = min(self.current_speed + 1.0, self.sumo.vehicle.getMaxSpeed(self.id))
        self.sumo.vehicle.setSpeed(self.id, new_speed)
        self.current_speed = new_speed

    def maintain(self):
        """
        Maintain the current speed.
        """
        self.sumo.vehicle.setSpeed(self.id, self.current_speed)

    def decelerate(self):
        """
        Decrease the vehicle's speed.
        """
        new_speed = max(self.current_speed - 1.0, 0)
        self.sumo.vehicle.setSpeed(self.id, new_speed)
        self.current_speed = new_speed
    
    def get_observation(self):
        position = self.sumo.vehicle.getPosition(self.id)
        speed = self.sumo.vehicle.getSpeed(self.id)
        acceleration = self.sumo.vehicle.getAcceleration(self.id)
        lane = self.sumo.vehicle.getLaneID(self.id)
        edge = self.sumo.vehicle.getRoadID(self.id)
        if edge is None or edge == '' or edge[0] == ':':
            edge = -1
        else:
            edge = int(edge) / 6
        observation = list(position) + [speed, acceleration, lane, edge]

        # 获取红绿灯状态信息
        for ts_id in self.env.ts_ids:
            traffic_signal_state = self.sumo.trafficlight.getRedYellowGreenState(ts_id)
            observation.append(traffic_signal_state)

        # 获取周围车辆信息
        surrounding_vehicles = self.sumo.vehicle.getNeighbors(self.id, self.num_surrounding_vehicles)
        headways = [1000] * self.num_surrounding_vehicles
        tailways = [1000] * self.num_surrounding_vehicles
        vel_in_front = [0] * self.num_surrounding_vehicles
        vel_behind = [0] * self.num_surrounding_vehicles

        for i, v in enumerate(surrounding_vehicles):
            headways[i] = self.sumo.vehicle.getLaneHeadway(self.id, v)
            tailways[i] = self.sumo.vehicle.getLaneTailway(self.id, v)
            vel_in_front[i] = self.sumo.vehicle.getSpeed(v)
            vel_behind[i] = self.sumo.vehicle.getSpeed(v)

        observation += headways + tailways + vel_in_front + vel_behind

        # 获取每条边上的平均速度和密度
        for edge in self.sumo.edge.getIDList():
            veh_ids = self.sumo.vehicle.getIDList()
            if len(veh_ids) > 0:
                avg_speed = sum(self.sumo.vehicle.getSpeed(veh) for veh in veh_ids) / len(veh_ids)
                density = len(veh_ids) / self.sumo.edge.getLength(edge)
                observation += [avg_speed, density]
            else:
                observation += [0, 0]

        observation = np.array(observation, dtype=np.float32)
        return observation
        
    
        
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

    