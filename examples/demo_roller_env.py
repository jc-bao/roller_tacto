# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import gym
import roller_env  # noqa: F401


class GraspingPolicy(torch.nn.Module):
  def __init__(self, env):
    super().__init__()
    self.env = env
    self.t = 0

  def forward(self, states=None):
    action = self.env.action_space.new()
    if not states:
      return action

    z_low, z_high = 0.05, 0.4
    dz = 0.02
    w_open, w_close = 0.05, 0.02
    gripper_force = 20

    # if self.t < 20:
    #   action.end_effector.position = states.object.position + [0, 0, z_high]
    #   action.end_effector.orientation = [0.0, 1, 0.0, 0.0]
    #   action.gripper_angle= w_open
    # elif self.t < 40:
    #   s = (self.t - 20) / 20
    #   z = z_high - s * (z_high - z_low)
    #   # action.end_effector.position = states.object.position + [0, 0, z]
    #   action.end_effector.position = states.object.position
    if self.t < 10:
      action.gripper_width = w_close
      action.gripper_force = gripper_force
    elif self.t < 20:
      delta = [0, 0, 0.1]
      action.end_effector.position = states.robot.end_effector.position + delta
      action.gripper_angle = w_close
      action.gripper_force = gripper_force
    elif self.t < 100:
      action.gripper_angle = w_close
      action.gripper_force = gripper_force
      action.pitch_l_angle = (self.t-80)/10
      action.pitch_r_angle = (self.t-80)/10
    elif self.t < 120:
      action.gripper_angle = w_close
      action.gripper_force = gripper_force
      action.roll_l_angle = (self.t-100)/10
      action.roll_r_angle = (self.t-100)/10
    else:
      action.gripper_width = w_close

    self.t += 1

    return action


def main():
  env = gym.make("roller-v0")
  print(f"Env observation space: {env.observation_space}")
  env.reset()

  # Create a hard-coded grasping policy
  policy = GraspingPolicy(env)

  # Set the initial state (obs) to None, done to False
  obs, done = None, False

  while not done:
    env.render()
    action = policy(obs)
    obs, reward, done, info = env.step(action)

  env.close()


if __name__ == "__main__":
  main()