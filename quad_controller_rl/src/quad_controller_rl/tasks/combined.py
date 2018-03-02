"""Combined task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class Combined(BaseTask):
    """Combined task of takeoff, hover and landing."""

    def __init__(self):
        # task name for saving models
        self.taskname = 'combined'

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
        self.max_duration = 15.0  # secs
        self.target_z = 10.0  # target height (z position) to reach for successful takeoff
        self.last_timestamp = None
        self.last_position = None
        self.velocity_weight = 0.5

    def reset(self):
        self.last_timestamp = None
        self.last_position = None
        return Pose(
                position=Point(0.0, 0.0, np.random.normal(0.5, 0.1)),  # drop off from a slight random height
                orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # Prepare state vector (pose only; ignore angular_velocity, linear_acceleration)
        state = np.array([
                pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])

        # get velocity
        if self.last_timestamp is None:
            velocity = 0.0
        else:
            velocity = abs(pose.position.z - self.last_position) / \
                        max(timestamp - self.last_timestamp, 1e-03)  # prevent divide by zero

        # scale elements to [0,1]
        scaled_z = pose.position.z / self.observation_space.high[2]
        max_v = (self.observation_space.high[2] - self.observation_space.low[2]) / 1e-03
        scaled_v = velocity / max_v
        scaled_t = timestamp / self.max_duration

        state = np.array([scaled_z, scaled_v, scaled_t])

        # compute reward
        done = False
        reward = 0.0
        # takeoff and hover
        if timestamp <= 10.0:
            reward += -abs(pose.position.z - self.target_z) # diff between position and target as penalty
        else:
            reward += -velocity / max(pose.position.z, 1e-03) # velocity as penalty bigger when close to land

        # update states
        self.last_timestamp = timestamp
        self.last_position = pose.position.z

        if timestamp > self.max_duration:  # task done
            reward += 1000. # survival bonus
            reward += -pose.position.z # z position as penalty(no penalty when landed)
            done = True
        elif pose.position.z > self.target_z + 10:
            reward += -1000. # task failed
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
