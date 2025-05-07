import math
import numpy as np

import gymnasium.spaces as spaces
from gymnasium.envs.registration import EnvSpec

from agxPythonModules.agxGym.agx_env import AGXGymEnv
from agxPythonModules.agxGym.utils import EnvironmentSceneDecorator, DisplayVirtualCameraGUI
from agxPythonModules.sensors.camera_sensors import VirtualCameraSensor

# AGX Dynamics imports
import agx
import agxOSG
import agxUtil
import agxRender
import agxCollide


class CartPoleEnv(AGXGymEnv):
    '''
    Models the cartpole AGXGym environment.

    Observation space is either a 4 scalars or a 64,64 rgb image.

    Action is the force to apply to the box at each timestep
    '''
    CART_CONFIG = {
        'width': 1.0,
        'height': 0.5,
        'depth': 0.5
    }
    POLE_CONFIG = {
        'length': 2.0,
        'width': 0.1,
        'thickness': 0.1,
    }

    angle_reward_limit = 0.2095
    pos_reward_limit = 2.4

    metadata = {"render_modes": ["human", "rgb_array"]}
    render_mode = None
    spec: EnvSpec = EnvSpec(id="agx-cartpole-v0", entry_point=None, max_episode_steps=200)

    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64)
    # The default observation space is [cart position, cart velocity, pole angle, pole angular velocity]
    # The environment can also have a camera and override the observation space
    observation_space = spaces.Box(
        low=np.array([-5.0, -10, -0.418, -6.0]),
        high=np.array([5.0, 10, 0.418, 6.0]),
        dtype=np.float64
    )

    def _build_scene(self):
        self.random_state = 0.2

        # Add Pole
        pole_geometry = agxCollide.Geometry(
            agxCollide.Box(
                self.POLE_CONFIG['width'] / 2,
                self.POLE_CONFIG['thickness'] / 2,
                self.POLE_CONFIG['length'] / 2
            )
        )
        self.pole = agx.RigidBody()
        self.pole.add(pole_geometry)
        length = self.POLE_CONFIG['length'] / 2

        # With random angle
        angle = self.np_random.uniform(-self.random_state, self.random_state)
        self.pole.setRotation(agx.Quat(angle, agx.Vec3().Y_AXIS()))
        self.pole.setPosition(agx.Vec3(length * math.sin(angle), 0, length * math.cos(angle)))
        self.sim.add(self.pole)

        # Add Cart
        self.cart = agx.RigidBody(agxCollide.Geometry(
            agxCollide.Box(
                self.CART_CONFIG['width'] / 2,
                self.CART_CONFIG['depth'] / 2,
                self.CART_CONFIG['height'] / 2)
        ))
        self.sim.add(self.cart)

        # Set cart track
        self.cart_track = agx.Prismatic(agx.Vec3.X_AXIS(), self.cart)
        self.sim.add(self.cart_track)

        agxUtil.setEnableCollisions(self.pole, self.cart, False)

        # Add Pole joint
        pole_frame = agx.Frame()
        pole_frame.setLocalTranslate(0, 0, -length)
        pole_frame.setLocalRotate(agx.Quat(agx.Vec3.Z_AXIS(), agx.Vec3.Y_AXIS()))
        cart_frame = agx.Frame()
        cart_frame.setLocalRotate(agx.Quat(agx.Vec3.Z_AXIS(), agx.Vec3.Y_AXIS()))

        self.pole_joint = agx.Hinge(self.pole, pole_frame, self.cart, cart_frame)
        self.sim.add(self.pole_joint)

        self.cart.getMassProperties().setMass(1.)
        self.pole.getMassProperties().setMass(0.8)

    def _modify_visuals(self, root):
        node = agxOSG.findGeometryNode(self.cart.getGeometries()[0], root)
        agxOSG.setDiffuseColor(node, agxRender.Color.Black())

        node = agxOSG.findGeometryNode(self.pole.getGeometries()[0], root)
        agxOSG.setDiffuseColor(node, agxRender.Color.OrangeRed())

        self.app.getSceneDecorator().setBackgroundColor(agxRender.Color.BlanchedAlmond(), agxRender.Color.DimGray())

        cameraData = self.app.getCameraData()
        cameraData.eye = agx.Vec3(0, -20, 3.0)
        cameraData.center = agx.Vec3(0, 0, 0)
        cameraData.up = agx.Vec3(0, 0, 1)
        cameraData.nearClippingPlane = 0.1
        cameraData.farClippingPlane = 5000
        self.app.applyCameraData(cameraData)

        self.app.getSceneDecorator().setEnableLogo(False)
        if self.render_mode != 'rgb_array' and self.environment_scene_decorator is None:
            self.environment_scene_decorator = EnvironmentSceneDecorator(self.app)

    def _setup_virtual_cameras(self, visual_observation_space: bool = False):
        # Creates one virtual camera sensor looking at the cart.
        camera = VirtualCameraSensor(
            64,
            64,
            self.cart.getPosition() - agx.Vec3(0, -2.5, -0.6),
            self.cart.getPosition() - agx.Vec3(0, 0, -0.6),
            agx.Vec3().Z_AXIS(),
            fovy=72,
            near=1.0,
            far=100.0,
            depth_camera=False)
        self.virtual_cameras.append(camera)

        # if the visual observation space is set to True change the self.observation_space of the environment
        if visual_observation_space:
            self.observation_space = spaces.Box(
                0, 255, shape=[64, 64, 3], dtype=np.uint8
            )

        # This is not headless. So I want a gui for viewing.
        if not self.headless:
            self.virtual_camera_gui = DisplayVirtualCameraGUI(self.virtual_cameras, 256, 256)

    def reset(self, *, seed=None, options=None):
        super(AGXGymEnv, self).reset(seed=seed)
        self.cart.setPosition(agx.Vec3())
        self.cart.setVelocity(agx.Vec3())
        self.cart.setAngularVelocity(agx.Vec3())

        self.pole.setVelocity(agx.Vec3())
        self.pole.setAngularVelocity(agx.Vec3())

        # With random angle
        length = self.POLE_CONFIG['length'] / 2
        angle = self.np_random.uniform(-self.random_state, self.random_state)
        self.pole.setRotation(agx.Quat(angle, agx.Vec3().Y_AXIS()))
        self.pole.setPosition(agx.Vec3(length * math.sin(angle), 0, length * math.cos(angle)))

        obs, _, _, _, d = self._observe()
        self.episode_step = 0
        self.rendered_this_step = False
        self.sim.setTimeStamp(0.0)

        return obs, d

    def _observe(self):
        # Are we using visual observations or not?
        visual = False
        if len(self.observation_space.shape) > 1:
            visual = True

        pole_omega = self.pole.getAngularVelocity()[1]
        cart_pos = self.cart.getPosition()[0]
        cart_velocity = self.cart.getVelocity()[0]
        pole_angle = self.pole_joint.getAngle()
        o = np.array([cart_pos, cart_velocity, pole_angle, pole_omega])
        terminal = self._terminal()
        reward = self._reward()
        truncated = False
        if self.spec.max_episode_steps is not None:
            truncated = self.episode_step >= self.spec.max_episode_steps

        if self.environment_scene_decorator is not None and self.render_mode == "human":
            self.environment_scene_decorator.update(self.episode_step, reward, observation=o)

        observation = o
        if visual and self.render_mode == "rgb_array":
            observation = self.render()
        observation = np.clip(observation, self.observation_space.low, self.observation_space.high)
        return observation, reward, terminal, truncated, {}

    def _set_action(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.cart.addForce(100 * float(action[0]), 0.0, 0.0)

    def _reward(self):
        pole_angle = self.pole_joint.getAngle()
        cart_pos = self.cart.getPosition()[0]
        r = 1
        # outside of these limits the reward is zero
        if abs(pole_angle) >= self.angle_reward_limit or abs(cart_pos) > self.pos_reward_limit:
            r = 0
        return r

    def _terminal(self):
        cart_pos = self.cart.getPosition()[0]
        pole_angle = self.pole_joint.getAngle()
        terminal = False
        # somewhat outside the reward limit the agents fails and the episode terminates
        if abs(cart_pos) > 0.2 + self.pos_reward_limit or abs(pole_angle) >= 1.5 * self.angle_reward_limit:
            terminal = True
        return terminal
