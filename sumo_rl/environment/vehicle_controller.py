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

    def __init__(self, sumo, env,vehicle_id="vehicle1"):
        self.vehicle_id=vehicle_id
        self.sumo = sumo
        self.env = env
        self.id = vehicle_id
        self.current_speed = self.sumo.vehicle.getSpeed(self.id)
        self.next_action_time = env.sim_step
        self.delta_time = 1  # 
        

    def set_next_action(self, action: str):
        """
        Sets the next action for the vehicle which could be accelerate, maintain, or decelerate.

        Args:
            action (str): One of ['accelerate', 'maintain', 'decelerate']
        """
        if self.env.sim_step < self.next_action_time:
            return  # 未到下一次行为设置时间

        if action == 'accelerate':
            self.accelerate()
        elif action == 'maintain':
            self.maintain()
        elif action == 'decelerate':
            self.decelerate()
        else:
            raise ValueError(f"Unknown action: {action}")

        self.next_action_time = self.env.sim_step + self.delta_time

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
        # 获取车辆的观察值
        # 根据你的需求返回合适的值，例如速度、位置等
        observation = {
            "speed": self.sumo.vehicle.getSpeed(self.id),
            # 添加其他需要的观测值
        }
        return observation
        
    def compute_reward(self):
        """
        Compute the reward for the vehicle.
        """
        # 示例奖励函数：根据车辆速度给予奖励，速度越高奖励越高
        speed = self.sumo.vehicle.getSpeed(self.id)
        waiting_time=self.sumo.vehicle.getWaitingTime(self.id)
        reward = -waiting_time
        return reward

    