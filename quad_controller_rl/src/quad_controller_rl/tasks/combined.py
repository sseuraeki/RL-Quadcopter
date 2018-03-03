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

        # Task-specific parameters
        self.max_duration = 15.0  # 5 secs for each task
        self.target_z = 10.0  # target height (z position)
        self.last_timestamp = None
        self.last_position = None
        self.position_weight = 0.7 # to adjust rewards mechanism
        self.velocity_weight = 0.3

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

        # set mode
        # 1 = takeoff, 2 = hover, 3 = landing
        if timestamp <= 5.0:
            self.mode = 1
        elif timestamp <= 10.0:
            self.mode = 2
        else:
            self.mode = 3

        # get velocity
        if self.last_timestamp is None:
            velocity = 0.0
        else:
            velocity = abs(pose.position.z - self.last_position) / \
                        max(timestamp - self.last_timestamp, 1e-03)  # prevent divide by zero

        # scale elements to [0,1]
        max_z = 300.
        min_z = 0.
        scaled_z = pose.position.z / max_z

        state = np.array([scaled_z, velocity])

        # compute reward
        done = False
        reward = 0.0
        # takeoff and hover rewards
        if timestamp <= 10.0:
            position_score = -abs(pose.position.z - self.target_z) # diff between position and target as penalty
            reward += self.position_weight * position_score
        else:
            position_score = -pose.position.z
            velocity_score = -velocity / max(pose.position.z, 1e-03) # velocity as penalty bigger when close to land
            reward += self.position_weight * position_score
            reward += self.velocity_weight * velocity_score

        # reset situations
        if timestamp > 5.0 and timestamp <= 10.0 \
           and abs(pose.position.z - self.target_z) > 1.0:
           reward += - 100. # penalty for not accomplishing takeoff
           done = True # reset
        elif pose.position.z > 15.0:
            reward += - 100. # penalty for going too high up
            done = True

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

        # longer it survives, more reward
        reward += timestamp * 10.

        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        action = self.agent.step(state, reward, done, self.mode)

        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = np.clip(action.flatten(), -25., 25.)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(0., 0., action),
                    torque=Vector3(0., 0., 0.)
                ), done
        else:
            return Wrench(), done
