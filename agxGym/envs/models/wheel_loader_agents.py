import agx
import agxSDK
import agxUtil
import agxCollide
import agxTerrain
import agxPowerLine
import agxDriveTrain

from agxPythonModules.models.wheel_loaders import WheelLoaderDL300, WheelLoaderWA475, WheelLoaderL70, WheelLoaderAlgoryx
from agxPythonModules.sensors.lidar_sensor import LidarSensor1D
from agxPythonModules.models.wheel_loaders import WheelLoader
from agxPythonModules.utils.environment import simulation

from .model_utils import reset_bodies, reset_constraints, reset_powerline

import math
import numpy as np
from enum import Enum

import logging
logger = logging.getLogger(__name__)


class LoaderAgent(agxSDK.Assembly):
    class BoomControl(Enum):
        VEL = 0
        FORCE = 1

    def __init__(self, **kwargs):
        super(LoaderAgent, self).__init__()
        self._local_transforms = []
        self._wp = agx.Vec3()
        self._wr = agx.Quat()

    def save_transforms(self):
        for rb in self.getRigidBodies():
            self._local_transforms.append(rb.getLocalTransform())
        self._wp = self.getPosition()
        self._wr = self.getRotation()

    def set_position_relative_body(self, rb: agx.RigidBody, pos: agx.Vec3):
        # Get current world transform of the rb
        T0 = rb.getTransform()
        # Get what the world transform should be
        T1 = agx.AffineMatrix4x4()
        T1.setRotate(T0.getRotate())
        T1.setTranslate(pos)
        # calculate the transform to take T0 to T1
        T = T0.inverse() * T1
        # transform the parent transform in the same way
        self.setTransform(T * self.getTransform())

    def observation_range(self):
        raise NotImplementedError

    def observe(self):
        raise NotImplementedError

    def set_action(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def configure_materials(self):
        raise NotImplementedError


class WheelLoaderAgent(WheelLoader, LoaderAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._max_brake_torque = 6000
        self._max_elevate_speed = 0.35
        self._max_tilt_speed = 0.35
        self._max_steer_speed = 1.0
        self._max_contraint_motor_force_range = {}

        self._lidar = LidarSensor1D(simulation(),
                                    self.front_body.getPosition() + agx.Vec3(0.0, 0.0, 1.3),
                                    agx.Vec3().X_AXIS(),
                                    2,
                                    math.pi / 4,
                                    10,
                                    rb_origin=self.front_body,
                                    draw_lines=True)
        simulation().add(self._lidar)
        self._observer_frame = agx.ObserverFrame(self.front_body, self.front_body.getTransform().inverse())
        self._observer_frame.setLocalPosition(agx.Vec3(0, -1.0, -0.85))

        control_of_boom = kwargs.get('control_of_boom', 'vel') == 'vel'
        self._control_of_boom = LoaderAgent.BoomControl.VEL
        if control_of_boom == 'force':
            self._control_of_boom = LoaderAgent.BoomControl.FORCE

        self._measures_energy = kwargs.get('measure_energy', False)
        if self._measures_energy:
            energy_manager = simulation().getEnergyManager()
            for c in self.elevate_prismatics + self.tilt_prismatics:
                energy_manager.add(c)
            agxDriveTrain.EnergyManager.add(self.engine, simulation())

        self._load_pin_constraints = []

        for c in self.tilt_prismatics + self.elevate_prismatics + [self.steering_hinge]:
            c.getRange1D().setEnable(True)

        self.save_transforms()

        self._colliding_bodies = [
            self.bucket_body,
            self.front_body,
            self.rear_body
        ] + [t.getTireRigidBody() for t in self.tires]

        # Create a convex around in the bucket geometry
        # It can be used as a sensor to determine which objects are inside the bucket
        vertices = agx.Vec3Vector()
        for geom in self.bucket_body.getGeometries():
            if geom.getShape().asTrimesh() is not None:
                v = geom.getShape().asTrimesh().getMeshData().getVertices()
                local_transform = geom.getLocalTransform()
                for vec3 in v:
                    vv = vec3 * local_transform
                    vertices.append(vv)
        scaling = agx.Matrix3x3(agx.Vec3(0.95))
        bucket_sensor_geom = agxCollide.Geometry(agxUtil.createConvex(vertices, scaling, agx.Vec3()))
        bucket_sensor_geom.setName("bucketSensorGeom")
        bucket_sensor_geom.setSensor(True)
        self.bucket_body.add(bucket_sensor_geom)
        self._bucket_sensor_geom = bucket_sensor_geom

        self._virtual_camera_eye = agx.Vec3(0, 2, 1)
        self._virtual_camera_center = agx.Vec3(0, -3, 2.0)
        self._virtual_camera_up = agx.Vec3().Z_AXIS()
        self._virtual_camera_ref_body = self.bucket_body

    def observation_range(self):
        delta = 0.1
        low = [
            -5,
            self.tilt_prismatics[0].getRange1D().getRange().lower() - delta,
            -self._max_tilt_speed,
            self.tilt_prismatics[0].getMotor1D().getForceRange().lower(),
            self.elevate_prismatics[0].getRange1D().getRange().lower() - delta,
            -self._max_elevate_speed,
            self.elevate_prismatics[0].getMotor1D().getForceRange().lower(),
            self.steering_hinge.getRange1D().getRange().lower() - delta,
            -self._max_steer_speed,
            self.steering_hinge.getMotor1D().getForceRange().lower(),
            0,
            0,
            0,
            0,
            -self._max_brake_torque,
            0
        ] + [0 for _ in self._lidar._rays_dict.keys()]
        upper = [
            5,
            self.tilt_prismatics[0].getRange1D().getRange().upper() + delta,
            self._max_tilt_speed,
            self.tilt_prismatics[0].getMotor1D().getForceRange().upper(),
            self.elevate_prismatics[0].getRange1D().getRange().upper() + delta,
            self._max_elevate_speed,
            self.elevate_prismatics[0].getMotor1D().getForceRange().upper(),
            self.steering_hinge.getRange1D().getRange().upper() + delta,
            self._max_steer_speed,
            self.steering_hinge.getMotor1D().getForceRange().upper(),
            1e7,
            6e7,
            6e7,
            8000,
            self._max_brake_torque,
            1
        ] + [self._lidar._max_length for _ in self._lidar._rays_dict.keys()]
        return np.asarray(low), np.asarray(upper)

    def observe(self):
        '''
            The state observation of the wheel loader

            The observations in order are:
            * 0 is the current velocity of the wheel loader. Measured relative the front body if gear == 1 and rear body if gear == 0. Meaured in m/s.
            * 1 is the current position of the hydraulic cylinder that tilts the bucket. Measured in meters deviated from the start position.
            * 2 is the current speed of the hydraulic cylinder that tilts the bucket. Measured in m/s.
            * 3 is the current force applied to the hydraulic cylinder tilts the bucket. Measured in N.
            * 4 is the current position of the one hydraulic cylinder that lifts the bucket.  Measured in meters deviated from the start position.
            * 5 is the current speed of the one hydraulic cylinder that lifts the bucket. Measured in m/s.
            * 6 is the current force applied to the one hydraulic cylinder lifts the bucket. Measured in N.
            * 7 is the current position of hinge that steers the vehicle.  Measured in radians deviated from the start position.
            * 8 is the current speed of hinge that steers the vehicle. Measured in rad/s
            * 9 is the current force applied to hinge steers the vehicle. Measured in N.
            * 10 is the magnitude of the force on load pin 1. That is one of the pins that connects the bucket to the boom. Measured in N.
            * 11 is the magnitude of the force on load pin 2. That is one of the pins that connects the bucket to the boom. Measured in N.
            * 12 is the magnitude of the force on load pin 3. That is one of the pins that connects the bucket to the boom. Measured in N.
            * 13 is the RPM of the combustion engine.
            * 14 is the current force applied to as braking. Measured in N.
            * 15 is the current gear of the gear box. This can only be 0 or 1. 0 is for reversing, 1 is for driving forward. Unitless.
            * 16-20 are lidar distances. Measured in meters from the cabin to the closest geometry it collides with.
        '''
        d = self._lidar.get_distances()
        o = []
        o.append(self.speed)
        o.append(self.tilt_prismatics[0].getAngle())  # The position of the prismatic
        o.append(self.tilt_prismatics[0].getCurrentSpeed())  # The actual current speed of the tilt prismatic
        o.append(self.tilt_prismatics[0].getMotor1D().getCurrentForce())  # The actual current force applied by the motor
        o.append(self.elevate_prismatics[0].getAngle())
        o.append(self.elevate_prismatics[0].getCurrentSpeed())
        o.append(self.elevate_prismatics[0].getMotor1D().getCurrentForce())
        o.append(self.steering_hinge.getAngle())
        o.append(self.steering_hinge.getCurrentSpeed())
        o.append(self.steering_hinge.getMotor1D().getCurrentForce())
        o.append(self.load_pin_1)  # The net force on the bucket from the hinge constraint
        o.append(self.load_pin_2)  # The net force on the bucket from the hinge constraint
        o.append(self.load_pin_3)  # The net force on the bucket from the hinge constraint
        # There are two prismatics that elvate the bucket.
        # These two will almost always be pretty same.
        # But with uneven load in the bucket they can ofcourse
        # vary a little.
        # self.elevate_prismatics[1].getAngle(),
        # self.elevate_prismatics[1].getCurrentSpeed(),
        o.append(self.engine.getRPM())  # The rpm of the combustion engine
        o.append(self.m_brake_hinge.getMotor1D().getCurrentForce())
        o.append(self.gear_box.getGear())  # This is discrete
        o = np.array(o + d)
        return o

    def action_range(self):
        low = np.asarray([-1] * 4)
        high = np.asarray([1] * 4)
        return low, high

    def set_action(self, action):
        self._engine_action_handler(action[0])
        # assumes value between -1, 1
        self._actuator_action_handler(action[1:])

    def reset(self, pos: agx.Vec3, forward: agx.Vec3):
        reset_bodies(self.getRigidBodies(), self._local_transforms)
        reset_constraints(self.getConstraints())
        reset_powerline(self.powerline)

        # restarts the engine and makes sure it is running
        self.engine.setEnable(False)
        self.engine.setEnable(True)

        self.setPosition(self._wp)
        self.setRotation(self._wr)

        rot = agx.Quat(self.front_forward_world, forward)
        self.setRotation(rot * self.getRotation())
        self.setPosition(pos)

        self.gear_box.setGear(1)
        self.engine.setThrottle(0)

    def create_shovel(self):
        shovel = super().create_shovel()
        # Turn off creating particles for the left and right side of the shovel.
        # It is not supposed to deform the terrain using its sides
        shovel.getExcavationSettings(agxTerrain.Shovel.ExcavationMode_DEFORM_LEFT).setEnableCreateDynamicMass(False)
        shovel.getExcavationSettings(agxTerrain.Shovel.ExcavationMode_DEFORM_RIGHT).setEnableCreateDynamicMass(False)
        shovel.getExcavationSettings(agxTerrain.Shovel.ExcavationMode_DEFORM_BACK).setEnableCreateDynamicMass(False)
        shovel.setPenetrationForceScaling(4)
        return shovel

    def configure_materials(self, terrain: agxTerrain.Terrain):
        # Set contact materials of the terrain and shovel
        # This contact material governs the resistance that the shovel will feel when digging into the terrain
        # [Shovel - Terrain] contact material
        shovel_material = self.bucket_body.getGeometries()[0].getMaterial()
        terrain_material = terrain.getMaterial(agxTerrain.Terrain.MaterialType_TERRAIN)
        shovel_terrain_contact_material = agx.ContactMaterial(shovel_material, terrain_material)
        shovel_terrain_contact_material.setYoungsModulus(1e8)
        shovel_terrain_contact_material.setRestitution(0.0)
        shovel_terrain_contact_material.setFrictionCoefficient(0.4)
        simulation().add(shovel_terrain_contact_material)

        # Reuse the contact material of particle-particle set by the material library
        # [Shovel - Particle] contact material
        particle_particle_cm = terrain.getContactMaterial(agxTerrain.Terrain.MaterialType_PARTICLE, agxTerrain.Terrain.MaterialType_PARTICLE)

        particle_mat = terrain.getMaterial(agxTerrain.Terrain.MaterialType_PARTICLE)
        shovel_particle_contact_material = agx.ContactMaterial(shovel_material, particle_mat)
        shovel_particle_contact_material.setYoungsModulus(particle_particle_cm.getYoungsModulus())
        shovel_particle_contact_material.setRestitution(particle_particle_cm.getRestitution())
        shovel_particle_contact_material.setFrictionCoefficient(particle_particle_cm.getFrictionCoefficient())
        shovel_particle_contact_material.setRollingResistanceCoefficient(particle_particle_cm.getRollingResistanceCoefficient())
        simulation().add(shovel_particle_contact_material)

        tire_materials = [tire.material for tire in self.tires]
        tire_ground_contact_materials = []
        for ground_material in [terrain_material]:
            for tire_material in tire_materials:
                tire_ground_cm = simulation().getMaterialManager().getOrCreateContactMaterial(tire_material, ground_material)
                tire_ground_cm.setYoungsModulus(1.0E5)
                tire_ground_cm.setFrictionModel(agx.ScaleBoxFrictionModel(agx.FrictionModel.DIRECT))
                tire_ground_cm.setFrictionCoefficient(1, agx.ContactMaterial.PRIMARY_DIRECTION)
                tire_ground_cm.setFrictionCoefficient(1, agx.ContactMaterial.SECONDARY_DIRECTION)
                tire_ground_contact_materials.append(tire_ground_cm)
        return tire_ground_contact_materials

    def _engine_action_handler(self, action: float):
        action = max(min(action, 1.0), -1.0)
        action = float(action)

        # Idle and close to zero speed. Engage the brakes.
        if abs(self.speed) < 0.15 and action == 0.0:
            self._set_brake(1.0)
            self._set_throttle(0.0)
        else:
            if action > 0:
                # Want to go forward. But we are going backwards. Brake
                if self.speed < -0.15:
                    self._set_throttle(0.0)
                    self._set_brake(action)
                else:
                    self.gear_box.setGear(1)
                    self._set_brake(0.0)
                    self._set_throttle(action)
            elif action < 0:
                if self.speed > 0.15:
                    self._set_throttle(0.0)
                    self._set_brake(abs(action))
                else:
                    self.gear_box.setGear(0)
                    self._set_brake(0)
                    self._set_throttle(abs(action))
            else:
                self._set_brake(0.0)
                self._set_throttle(0.0)

    def _actuator_action_handler(self, actions: np.array):
        for c in self.tilt_prismatics:
            if self._control_of_boom == LoaderAgent.BoomControl.VEL:
                self._set_motor_speed(c, float(self._max_tilt_speed * actions[0]))
            elif self._control_of_boom == LoaderAgent.BoomControl.FORCE:
                self._set_motor_force(c, float(self._max_contraint_motor_force_range[c.getName()] * actions[0]))
        for c in self.elevate_prismatics:
            if self._control_of_boom == LoaderAgent.BoomControl.VEL:
                self._set_motor_speed(c, float(self._max_elevate_speed * actions[1]))
            elif self._control_of_boom == LoaderAgent.BoomControl.FORCE:
                self._set_motor_force(c, float(self._max_contraint_motor_force_range[c.getName()] * actions[1]))
        if self._control_of_boom == LoaderAgent.BoomControl.VEL:
            self._set_motor_speed(self.steering_hinge, float(self._max_steer_speed * actions[2]))
        elif self._control_of_boom == LoaderAgent.BoomControl.FORCE:
            self._set_motor_force(c, float(self._max_contraint_motor_force_range[c.getName()] * actions[2]))

    def _set_motor_speed(self, constraint: agx.Constraint, speed: float):
        speed = float(speed)
        constraint.getLock1D().setEnable(False)
        constraint.getMotor1D().setLockedAtZeroSpeed(True)
        constraint.getMotor1D().setEnable(True)
        constraint.getMotor1D().setSpeed(speed)

    def _set_motor_force(self, constraint: agx.Constraint, force: float):
        force = float(force)
        constraint.getLock1D().setEnable(False)
        constraint.getMotor1D().setEnable(True)
        constraint.getMotor1D().setSpeed(0.0)
        constraint.getMotor1D().setForceRange(force, force)

    def _set_throttle(self, value: float):
        # The throttle can only be between 0, 1
        self.engine.setThrottle(value)

    def _set_brake(self, value: float):
        brake_torque = value * self._max_brake_torque
        self.m_brake_hinge.setEnable(True)
        self.m_brake_hinge.getMotor1D().setEnable(value > 0)
        self.m_brake_hinge.getMotor1D().setSpeed(0)
        self.m_brake_hinge.getMotor1D().setForceRange(-brake_torque, brake_torque)

    @property
    def energy_consumption(self):
        constraint_power = 0
        engine_power = 0
        if self._measures_energy:
            energy_manager = simulation().getEnergyManager()
            # Clamping the power output to positive. Because combustion engine wheel loaders cannot
            # save or retrieve energy from rolling down hills etc. Like an eletric wheel loader
            # would be able to do with regenerative braking.
            for c in self.elevate_prismatics + self.tilt_prismatics:
                constraint_power += max(energy_manager.getPower(c.getMotor1D()), 0)
            engine_power += max(agxDriveTrain.EnergyManager.getPower(self.engine), 0)
        return constraint_power + engine_power

    @property
    def load_pin_1(self):
        try:
            force = agx.Vec3()
            torque = agx.Vec3()
            self._load_pin_constraints[0].getLastForce(self.bucket_body, force, torque, True)
        except IndexError:
            logging.ERROR("The inheriting model must get the 3 load pin constraints by name")
        return force.length()

    @property
    def load_pin_2(self):
        try:
            force = agx.Vec3()
            torque = agx.Vec3()
            self._load_pin_constraints[1].getLastForce(self.bucket_body, force, torque, True)
        except IndexError:
            logging.ERROR("The inheriting model must get the 3 load pin constraints by name")
        return force.length()

    @property
    def load_pin_3(self):
        try:
            force = agx.Vec3()
            torque = agx.Vec3()
            self._load_pin_constraints[2].getLastForce(self.bucket_body, force, torque, True)
        except IndexError:
            logging.ERROR("The inheriting model must get the 3 load pin constraints by name")
        return force.length()

    @property
    def colliding_bodies(self):
        return self._colliding_bodies

    @property
    def virtual_camera_eye(self) -> agx.Vec3:
        return self._virtual_camera_eye

    @property
    def virtual_camera_center(self) -> agx.Vec3:
        return self._virtual_camera_center

    @property
    def virtual_camera_up(self) -> agx.Vec3:
        return self._virtual_camera_up

    @property
    def virtual_camera_ref_body(self) -> agx.RigidBody:
        return self._virtual_camera_ref_body


class WheelLoaderDL300Agent(WheelLoaderDL300, WheelLoaderAgent):
    def __init__(self, **kwargs):
        super(WheelLoaderDL300Agent, self).__init__(**kwargs)

        self.bucket_tilt_controller = None

        # Enable tilt prismatic and set Range so that it it not possible to drive up onto shovel
        self.tilt_prismatics[0].getRange1D().setEnable(True)
        self.tilt_prismatics[0].getRange1D().setRange(-1, 0.30)

        self._load_pin_constraints = [
            self.getConstraint("Hinge9").asHinge(),
            self.getConstraint("TrackedRangeHinge2").asHinge(),
            self.getConstraint("TrackedRangeHinge3").asHinge()
        ]
        for c in self._load_pin_constraints:
            c.setEnableComputeForces(True)

    def heuristic_control_policy(self, t):
        if t < 2.2:
            return np.array([0.0, .2, -1.0, 0.0])
        # Drive forward
        elif t >= 2.2 and t < 5.0:
            return np.array([0.4, 0.0, 0.0, 0.0])
        # Raise bucket
        elif t >= 5.0 and t < 6.0:
            return np.array([0.4, -0.2, 0.5, 0.0])
        # Back away
        elif t >= 6.5 and t < 9.0:
            return np.array([-0.1, -0.2, 0.0, 0.0])
        else:
            return np.array([0.0, 0.0, 0, 0.0])


class WheelLoaderWA475Agent(WheelLoaderWA475, WheelLoaderAgent):
    def __init__(self, **kwargs):
        super(WheelLoaderWA475Agent, self).__init__(**kwargs)
        self.bucket_tilt_controller = None
        self._max_brake_torque = 15000
        self._load_pin_constraints = [
            self.getConstraint("Hinge5").asHinge(),
            self.getConstraint("TrackedRangeHinge2").asHinge(),
            self.getConstraint("TrackedRangeHinge3").asHinge()
        ]
        for c in self._load_pin_constraints:
            c.setEnableComputeForces(True)

        self._lift_hinge1 = self.getConstraint("Hinge3").asHinge()
        self._lift_hinge2 = self.getConstraint("Hinge6").asHinge()
        self._lift_hinge1.getRange1D().setEnable(True)
        self._lift_hinge1.getRange1D().setRange(-1.5, 0.1)
        self._lift_hinge2.getRange1D().setEnable(True)
        self._lift_hinge2.getRange1D().setRange(-0.1, 1.5)

    @property
    def bucket_sensor_geom(self) -> agxCollide.Geometry:
        return self._bucket_sensor_geom

    def heuristic_control_policy(self, t):
        if t < 3.0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        elif t >= 3.0 and t < 5.0:
            return np.array([1.0, -0.2, 0.4, 0.0])
        elif t >= 5.0 and t < 5.5:
            return np.array([-1.0, 0.0, 0.45, -0.0])
        elif t >= 5.5 and t < 6.0:
            return np.array([-1.0, 0.0, 0.6, -0.7])
        elif t >= 6.0 and t < 9.5:
            return np.array([-0.7, 0.0, 0.0, -0.0])
        elif t >= 9.5 and t < 10.5:
            return np.array([1.0, 0.0, 0.0, 0.7])
        elif t >= 10.5 and t < 14.2:
            return np.array([1.0, 0.0, 0.0, 0.0])
        else:
            if abs(self.speed) > 0.1:
                return np.array([-np.sign(self.speed) * 0.43, 0.0, 0.0, 0.0])
            else:
                return np.array([0.0, 0.3, 0.3, 0.0])


class WheelLoaderL70Agent(WheelLoaderL70, WheelLoaderAgent):
    def __init__(self, **kwargs):
        super(WheelLoaderL70Agent, self).__init__(**kwargs)
        self.bucket_tilt_controller = None
        self._load_pin_constraints = [
            self.getConstraint("TrackedRangeHinge1").asHinge(),
            self.getConstraint("TrackedRangeHinge2").asHinge(),
            self.getConstraint("TrackedRangeHinge3").asHinge()
        ]
        for c in self._load_pin_constraints:
            c.setEnableComputeForces(True)

    @property
    def bucket_sensor_geom(self) -> agxCollide.Geometry:
        return self._bucket_sensor_geom

    def heuristic_control_policy(self, t):
        # Align bucket
        if t < 2.0:
            return np.array([0.0, 0.0, -0.1, 0.0])
        # Drive forward
        elif t >= 2.0 and t < 5.0:
            return np.array([0.8, 0.0, 0.0, 0.0])
        # Raise bucket
        elif t >= 5.0 and t < 6.0:
            return np.array([0.4, -0.2, 0.5, 0.0])
        # Back away
        elif t >= 6.5 and t < 9.0:
            return np.array([-0.4, -0.2, 0.0, 0.0])
        else:
            return np.array([0.0, 0.0, 0, 0.0])


class WheelLoaderAlgoryxAgent(WheelLoaderAlgoryx, WheelLoaderAgent):
    def __init__(self, **kwargs):
        super(WheelLoaderAlgoryxAgent, self).__init__(**kwargs)
        self.bucket_tilt_controller = None
        self._max_brake_torque = 3000
        self._max_elevate_speed = 0.2
        self._max_tilt_speed = 0.2
        self._load_pin_constraints = [
            self.getConstraint("TrackedRangeHinge1").asHinge(),
            self.getConstraint("TrackedRangeHinge2").asHinge(),
            self.getConstraint("TrackedRangeHinge3").asHinge()
        ]
        for c in self._load_pin_constraints:
            c.setEnableComputeForces(True)

        self._virtual_camera_eye = agx.Vec3(0, 0, 2)
        self._virtual_camera_center = agx.Vec3(2, 0, 1.0)
        self._virtual_camera_up = agx.Vec3().Z_AXIS()
        self._virtual_camera_ref_body = self.rear_body

    @property
    def bucket_sensor_geom(self) -> agxCollide.Geometry:
        return self._bucket_sensor_geom

    def heuristic_control_policy(self, t):
        # Drive forward
        if t >= 0.0 and t < 5.0:
            return np.array([0.75, 0.0, 0.0, 0.0])
        # Raise bucket
        elif t >= 5.0 and t < 5.6:
            return np.array([0.4, -0.2, 0.5, 0.0])
        # Back away
        elif t >= 6.0 and t < 9.0:
            return np.array([-0.7, -0.2, 0.0, 0.0])
        else:
            return np.array([0.0, 0.0, 0, 0.0])
