# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import cv2
import hydra
import pybullet as p
import pybulletX as px
import tacto  # Import TACTO
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
import scipy.spatial.transform as tf

log = logging.getLogger(__name__)


# Load the config YAML file from examples/conf/digit.yaml
@hydra.main(config_path="conf", config_name="digit")
def main(cfg):
  # Initialize digits
  bg = cv2.imread("conf/bg_digit_240_320.jpg")
  digits = tacto.Sensor(**cfg.tacto, background=bg)

  # Initialize World
  px.init()
  p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
  p.resetDebugVisualizerCamera(**cfg.pybullet_camera)
  p.setGravity(0, 0, 0)

  # Create and initialize DIGIT
  digit_body = px.Body(**cfg.digit)
  digits.add_camera(digit_body.id, [-1])

  # Add object to pybullet and tacto simulator
  obj = px.Body(**cfg.object)
  digits.add_body(obj)

  # Create control panel to control the 6DoF pose of the object
  # panel = px.gui.PoseControlPanel(obj, **cfg.object_control_panel)
  # panel.start()
  # log.info("Use the slides to move the object until in contact with the DIGIT")

  # run p.stepSimulation in another thread
  # t = px.utils.SimulationThread(real_time_factor=1.0)
  # t.start()

  depth_imgs = []
  for i in range(120):
    p.stepSimulation()
    color, depth = digits.render()
    theta = (2 * np.pi * i / 60) % (2 * np.pi)
    camera_z = (np.cos(theta % (np.pi / 2) - np.pi / 4) * 0.707) * 0.008
    obj_orn = [np.cos(theta / 2), 0, 0, np.sin(theta / 2)]
    obj.set_base_pose([-0.015, 0, 0.025 + (np.cos(theta % (np.pi / 2) - np.pi / 4) * 0.707 - 0.5) * 0.008], obj_orn)
    digits.updateGUI(color, depth)
    local_points = depth[0]
    img_size = local_points.shape
    xx = np.tile(np.arange(img_size[0]) / img_size[0] - 0.5, (img_size[1], 1)).transpose() * 0.005
    yy = np.tile(np.arange(img_size[1]) / img_size[0] - 0.5 * img_size[1] / img_size[0], (img_size[0], 1)) * 0.005
    local_points = np.stack((xx, yy, local_points), axis=-1)
    local_points = local_points[local_points[..., -1] > 0]
    if len(local_points) > 0:
      local_points[2] += camera_z
      camera_orn = tf.Rotation.from_quat(obj_orn)
      global_pos = camera_orn.apply(-local_points)
      depth_imgs.append(global_pos)
  for i, img in enumerate(depth_imgs):
    # cloud = PyntCloud(pd.DataFrame(
    #   data=img.reshape(-1, 3),
    #   columns=["x", "y", "z"]))
    # cloud.to_file(f"data/{i}.ply")
    np.savetxt(f"data/{i}.csv", img.reshape(-1, 3), delimiter=",")


if __name__ == "__main__":
  main()
