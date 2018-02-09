"""Takeoff task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class Hover(BaseTask):
    """the goal is to stay at 10 units height for a few seconds"""

    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,       0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([  cube_size / 2,   cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0]))
        #print("Takeoff(): observation_space = {}".format(self.observation_space))  # [debug]

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 25.0
        max_torque = 25.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))
        #print("Takeoff(): action_space = {}".format(self.action_space))  # [debug]

        # Task-specific parameters
        self.max_duration = 10.0  # secs
        self.target_z = 10.0  # target height (z position) to reach for successful takeoff

    def reset(self):
        # Nothing to reset; just return initial condition
        return Pose(
                position=Point(np.random.uniform(- 150., 150.),
                               np.random.uniform(- 150., 150.),
                               np.random.normal(10.0, 0.1)),  # start somewhere around 10 z
                orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
            ), Twist(
                linear=Vector3(np.random.normal(0., 1.0),
                               np.random.normal(0., 1.0),
                               np.random.normal(0., 1.0)), # after takeoff, there might be some velocity
                angular=Vector3(0.0, 0.0, 0.0)
            )

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # Prepare state vector (pose only; ignore angular_velocity, linear_acceleration)
        state = np.array([
                pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])

        # Compute reward / penalty and check if this episode is complete
        done = False
        reward = 0.0
        reward += -min(abs(self.target_z - pose.position.z), 20.0)  # the farther away from the target z, the less reward
        reward += timestamp # the longer the copter survives, the more reward

        # define done conditions
        if pose.position.z < 1.0:
            reward -= 100.0 # big penalty when it hits the ground (or very close to it)
            done = True
        elif pose.position.z > self.target_z + 10.0:
            reward -= 100.0 # big penalty when it goes too high up
            done = True
        elif timestamp > self.max_duration:  # task end
            # very big bonus when it survives the target duration
            # for debugging purposes, very big total_reward will indicate whether
            # the agent succeeded or not
            reward += 1000.0
            done = True

        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        action = self.agent.step(state, reward, done)  # note: action = <force; torque> vector

        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:
            return Wrench(), done
