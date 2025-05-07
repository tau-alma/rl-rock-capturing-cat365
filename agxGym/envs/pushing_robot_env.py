import gymnasium.spaces as spaces
from gymnasium.envs.registration import EnvSpec
import numpy as np

from agxPythonModules.agxGym.agx_env import AGXGymEnv
from agxPythonModules.robots.generic_robot import GenericRobot
from agxPythonModules.agxGym.utils import EnvironmentSceneDecorator

# AGX Dynamics imports
import agx
import agxCollide
import agxPowerLine
import agxDriveTrain
import agxOSG
import agxRender
import agxUtil


class RobotJoint():
    '''
    Help class created to save the hinge and its engine lookup table in the same place, and make it easier to set the
    torque applied to the hinge.
    '''

    def __init__(self, hinge, engineLookupTable):
        self.hinge = hinge
        self.lookupTable = engineLookupTable

    def getAngle(self):
        return self.hinge.getAngle()

    def setTorque(self, torque):
        self.lookupTable.clear()
        self.lookupTable.insertValue(0, torque)


class PushingRobotEnv(AGXGymEnv):
    '''
    Models a robot AGXGym environment. The robot has 2 degrees of freedom. And has a box on a rail in front of it.
    The box is placed at different heights relative to the robot. The task of the robot is to find the box
    and push it as far away from as possible.

    Observation space is the internal state of the robot and the relative distance from the tool tip to the box.

    Action is the force to apply to each joint at each time step.
    '''
    hinge_names = [
        "bottom",
        "bottomMiddle",
        "middleTop",
        "topHead",
        "head",
        "headPlate"
    ]
    hinge_force_ranges = [
        550,
        450,
    ]

    action_hinge_names = [
        "bottomMiddle",
        "middleTop"
    ]

    # List of values for motor inertia and gear ratio.
    motor_inertia = [
        0.046,
        0.036,
    ]
    gear_ratio = [
        171.0,
        143.0,
    ]
    z_pos_range_cart = [1.4, 2.2]

    time_step = 0.005

    metadata = {"render_modes": ["human"]}
    render_mode = None
    spec: EnvSpec = EnvSpec(id="agx-pushing-robot-v0", entry_point=None, max_episode_steps=300)

    action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64)
    observation_space = spaces.Box(
        low=np.asarray([-3.05, -5.0, -0.04, -5.0, -2.0, -2.0, -2.0, -2.0]),
        high=np.asarray([0.32, 5.0, 3.52, 5.0, 3.0, 2.0, 2.0, 2.0]),
        dtype=np.float64)

    def _build_scene(self):
        # Create the robot
        self.robot = GenericRobot(self.sim, create_visual=False)
        bottom_plate = self.robot.bodies["plate"]
        bottom_plate.setMotionControl(agx.RigidBody.STATIC)
        self.hinges = [self.robot.hinges[name] for name in self.hinge_names]

        for h in self.hinges:
            h.getMotor1D().setEnable(True)
            h.getMotor1D().setLockedAtZeroSpeed(True)
            h.getMotor1D().setSpeed(0)

        self.action_hinges = [self.robot.hinges[name] for name in self.action_hinge_names]

        # Add a engines and drivtrain to the two actuated joints
        powerline = agxPowerLine.PowerLine()
        self.sim.add(powerline)

        self.joints = []
        for i, hinge in enumerate(self.action_hinges):
            hinge.getMotor1D().setEnable(False)

            engine = agxDriveTrain.Engine()
            engine.setPowerGenerator(agxPowerLine.TorqueGenerator(engine.getRotationalDimension()))
            lookupTable = engine.getPowerGenerator().getPowerTimeIntegralLookupTable()
            engine.setThrottle(1.0)
            engine.setInertia(self.motor_inertia[i])
            powerline.add(engine)

            actuator = agxPowerLine.RotationalActuator(hinge)
            gear = agxDriveTrain.HolonomicGear()
            gear.setGearRatio(self.gear_ratio[i])

            engine.connect(gear)
            gear.connect(actuator)

            self.joints.append(RobotJoint(hinge, lookupTable))

        # Create sphere as tool tip
        r = 0.02
        self.tool = agx.RigidBody(agxCollide.Geometry(agxCollide.Sphere(r)))
        self.tool.setPosition(1.36, 0.085, 1.801)
        self.sim.add(self.tool)

        agxUtil.setEnableCollisions(self.robot.bodies["headPlate"], self.tool, False)

        f1 = agx.Frame()
        f2 = agx.Frame()
        agx.Constraint.calculateFramesFromWorld(self.tool.getPosition(), agx.Vec3.X_AXIS(), self.tool, f1, self.robot.bodies["headPlate"], f2)
        joint = agx.LockJoint(self.tool, f1, self.robot.bodies["headPlate"], f2)
        self.sim.add(joint)

        # Create a cart on a prismatic that the robot can push. And that tries to push back.
        self.cart = agx.RigidBody(agxCollide.Geometry(agxCollide.Box(agx.Vec3(0.2))))
        z_pos = np.random.uniform(self.z_pos_range_cart[0], self.z_pos_range_cart[1])
        self.cart.setPosition(agx.Vec3(1.8, 0.1, z_pos))
        self.sim.add(self.cart)
        self.cart_track = agx.Prismatic(agx.Vec3().X_AXIS(), self.cart)
        cart_track_range = self.cart_track.getRange1D()
        cart_track_range.setEnable(True)
        cart_track_range.setRange(agx.RangeReal(0.0, 1.5))
        cart_track_motor = self.cart_track.getMotor1D()
        cart_track_motor.setEnable(True)
        cart_track_motor.setSpeed(-1.0)
        cart_force = 2000
        cart_track_motor.setForceRange(-cart_force, cart_force)
        self.sim.add(self.cart_track)

        self.sim.setTimeStep(self.time_step)

    def _modify_visuals(self, root):
        node = agxOSG.findGeometryNode(self.cart.getGeometries()[0], root)
        agxOSG.setDiffuseColor(node, agxRender.Color.Black())

        self.app.getSceneDecorator().setBackgroundColor(agxRender.Color.BlanchedAlmond(), agxRender.Color.DimGray())

        cameraData = self.app.getCameraData()
        cameraData.eye = agx.Vec3(1.0, -7, 2.0)
        cameraData.center = agx.Vec3(1.0, 0, 1.0)
        cameraData.up = agx.Vec3(0, 0, 1)
        cameraData.nearClippingPlane = 0.1
        cameraData.farClippingPlane = 100
        self.app.applyCameraData(cameraData)

        self.app.getSceneDecorator().setEnableLogo(False)
        if self.render_mode != 'rgb_array' and self.environment_scene_decorator is None:
            self.environment_scene_decorator = EnvironmentSceneDecorator(self.app)

    def _observe(self):
        o = []
        for h in self.action_hinges:
            o.append(h.getAngle())
            o.append(h.getCurrentSpeed())
        tool_pos = self.tool.getPosition()
        box_pos = self.cart.getPosition()
        o.extend([tool_pos.x(), tool_pos.z(), tool_pos.x() - box_pos.x(), tool_pos.z() - box_pos.z()])
        o = np.array(o)

        terminal = self._terminal()
        r = self._reward()
        truncated = False
        if self.spec.max_episode_steps is not None:
            truncated = self.episode_step >= self.spec.max_episode_steps

        if self.environment_scene_decorator is not None and self.render_mode == "human":
            self.environment_scene_decorator.update(self.episode_step, r, observation=o)

        return o, r, terminal, truncated, {}

    def _set_action(self, action):
        for j, fr, a in zip(self.joints, self.hinge_force_ranges, action):
            self._set_force_range(j, fr * a)

    def _set_force_range(self, h, f):
        h.setTorque(f)

    def _collision_sensor(self):
        space = self.sim.getSpace()
        matches = agxCollide.GeometryContactPtrVector()
        nb_contacts = space.getGeometryContacts(matches, self.robot.bodies["headPlate"], self.cart)
        if nb_contacts > 0:
            return True
        return False

    def _reward(self):
        tool_pos = self.tool.getPosition()
        r = 0
        # Only give reward if the tool tip is in contact with casrt
        if self._collision_sensor():
            r = self.cart_track.getAngle()
        elif tool_pos.z() < 0.0:
            r = -1
        return r

    def _terminal(self):
        tool_pos = self.tool.getPosition()
        return tool_pos.z() < 0.0
