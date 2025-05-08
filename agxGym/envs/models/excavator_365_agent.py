import agx
import agxUtil
import agxCollide
import agxTerrain
import random
import agxModel

from agxPythonModules.models.excavators.excavator365 import Excavator365
from agxPythonModules.utils.environment import simulation

from .model_utils import reset_bodies, reset_constraints

from agxPythonModules.utils.callbacks import StepEventCallback

import numpy as np
import agxSDK
  
class Excavator365Agent(Excavator365):
    _max_arm_speed = 0.3
    _max_stick_speed = 0.3
    _max_bucket_speed = 0.2
    _max_track_speed = 3.0
    _max_rotate_speed = 0.3

    def __init__(self, **kwargs):
        super(Excavator365Agent, self).__init__()

        self._local_transforms = []
        self._track_node_transforms = []
        self._wp = agx.Vec3()
        self._wr = agx.Quat()

        self._bodies = []
        for rb in self.getRigidBodies():
            # print(rb.getName())
            self._bodies.append(rb)
        for rb in self.track(Excavator365.Location.LEFT).m_track.getRigidBodies():
            self._bodies.append(rb)
        for rb in self.track(Excavator365.Location.RIGHT).m_track.getRigidBodies():
            self._bodies.append(rb)

        self.save_transforms()
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

        # adjust force ranges
        for p in self.arm_prismatics:
            p.getMotor1D().setForceRange(-1e7, 1e7)
        self.cabin_hinge.getMotor1D().setForceRange(-3e6, 3e6)
        
        #######################################################
        # Create a collection to reconfigure the excavator
        self._col = agxSDK.Collection()
        self._col.add(self)
        self._ref_body = self.getRigidBody("ChassieBody")
        self._randomize_pose = RandomizePose(self._col, self._ref_body)
        
    def save_transforms(self):
        for rb in self._bodies:
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
        low = [
            self.arm_prismatics[0].getRange1D().getRange().lower(),
            -self._max_arm_speed,
            self.arm_prismatics[0].getMotor1D().getForceRange().lower(),
            self.stick_prismatic.getRange1D().getRange().lower(),
            -self._max_stick_speed,
            self.stick_prismatic.getMotor1D().getForceRange().lower(),
            self.bucket_prismatic.getRange1D().getRange().lower(),
            -self._max_bucket_speed,
            self.bucket_prismatic.getMotor1D().getForceRange().lower()
        ]
        upper = [
            self.arm_prismatics[0].getRange1D().getRange().upper(),
            self._max_arm_speed,
            self.arm_prismatics[0].getMotor1D().getForceRange().upper(),
            self.stick_prismatic.getRange1D().getRange().upper(),
            self._max_stick_speed,
            self.stick_prismatic.getMotor1D().getForceRange().upper(),
            self.bucket_prismatic.getRange1D().getRange().upper(),
            self._max_bucket_speed,
            self.bucket_prismatic.getMotor1D().getForceRange().upper()
        ]
        return np.asarray(low), np.asarray(upper)

    def observation_range_moving(self):
        low = [
            -2,
            self.arm_prismatics[0].getRange1D().getRange().lower(),
            -self._max_arm_speed,
            self.arm_prismatics[0].getMotor1D().getForceRange().lower(),
            self.stick_prismatic.getRange1D().getRange().lower(),
            -self._max_stick_speed,
            self.stick_prismatic.getMotor1D().getForceRange().lower(),
            self.bucket_prismatic.getRange1D().getRange().lower(),
            -self._max_bucket_speed,
            self.bucket_prismatic.getMotor1D().getForceRange().lower(),
            -self._max_rotate_speed,
            -1,
            -1,
            -1
        ]
        upper = [
            2,
            self.arm_prismatics[0].getRange1D().getRange().upper(),
            self._max_arm_speed,
            self.arm_prismatics[0].getMotor1D().getForceRange().upper(),
            self.stick_prismatic.getRange1D().getRange().upper(),
            self._max_stick_speed,
            self.stick_prismatic.getMotor1D().getForceRange().upper(),
            self.bucket_prismatic.getRange1D().getRange().upper(),
            self._max_bucket_speed,
            self.bucket_prismatic.getMotor1D().getForceRange().upper(),
            self._max_rotate_speed,
            1,
            1,
            1
        ]
        return np.asarray(low), np.asarray(upper)
    
    def action_range(self):
        low = np.asarray([-1] * 6)
        high = np.asarray([1] * 6)
        return low, high
    
    def action_range_bucket_arm_boom(self):
        low = np.asarray([-1] * 3)
        high = np.asarray([1] * 3)
        return low, high

    def observe(self):
        o = []
        o.append(self.arm_prismatics[0].getAngle())  # The position of the prismatic
        o.append(self.arm_prismatics[0].getCurrentSpeed())  # The actual current speed of the tilt prismatic
        o.append(self.arm_prismatics[0].getMotor1D().getCurrentForce())  # The actual current force applied by the motor
        o.append(self.stick_prismatic.getAngle())
        o.append(self.stick_prismatic.getCurrentSpeed())
        o.append(self.stick_prismatic.getMotor1D().getCurrentForce())
        o.append(self.bucket_prismatic.getAngle())
        o.append(self.bucket_prismatic.getCurrentSpeed())
        o.append(self.bucket_prismatic.getMotor1D().getCurrentForce())
        o = np.asarray(o)
        return o
    
    def observe_moving(self):
        o = []
        o.append(self.speed)
        o.append(self.arm_prismatics[0].getAngle())  # The position of the prismatic
        o.append(self.arm_prismatics[0].getCurrentSpeed())  # The actual current speed of the tilt prismatic
        o.append(self.arm_prismatics[0].getMotor1D().getCurrentForce())  # The actual current force applied by the motor
        o.append(self.stick_prismatic.getAngle())
        o.append(self.stick_prismatic.getCurrentSpeed())
        o.append(self.stick_prismatic.getMotor1D().getCurrentForce())
        o.append(self.bucket_prismatic.getAngle())
        o.append(self.bucket_prismatic.getCurrentSpeed())
        o.append(self.bucket_prismatic.getMotor1D().getCurrentForce())
        o.append(self.cabin_hinge.getCurrentSpeed())
        o.extend([*self.front_forward_world])
        o = np.asarray(o)
        return o
    
    def _set_motor_speed(self, constraint: agx.Constraint, speed: float):
        speed = float(speed)
        constraint.getLock1D().setEnable(False)
        constraint.getMotor1D().setLockedAtZeroSpeed(True)
        constraint.getMotor1D().setEnable(True)
        constraint.getMotor1D().setSpeed(speed)

    def set_action(self, action):
        for c in self.arm_prismatics:
            self._set_motor_speed(c, self._max_arm_speed * action[0])
        self._set_motor_speed(self.stick_prismatic, self._max_stick_speed * action[1])
        self._set_motor_speed(self.bucket_prismatic, self._max_bucket_speed * action[2])
        self._set_motor_speed(self.sprocket_hinge(self.Location.LEFT), self._max_track_speed * action[3])
        self._set_motor_speed(self.sprocket_hinge(self.Location.RIGHT), self._max_track_speed * action[4])
        self._set_motor_speed(self.cabin_hinge, self._max_rotate_speed * action[5])
    
    def set_action_bucket_arm_boom(self, action):
        for c in self.arm_prismatics:
            self._set_motor_speed(c, self._max_arm_speed * action[0])
        self._set_motor_speed(self.stick_prismatic, self._max_stick_speed * action[1])
        self._set_motor_speed(self.bucket_prismatic, self._max_bucket_speed * action[2])

    def reset(self, pos: agx.Vec3, config: agx.Vec3):
        reset_bodies(self._bodies, self._local_transforms)
        reset_constraints(self.getConstraints())
        self.setPosition(self._wp)
        self.setRotation(self._wr)
        self.setPosition(pos)
        
        
        arm = config[0]
        stick = config[1]
        bucket = config[2]
        
        bucket_angle_min = bucket
        bucket_angle_max = bucket
        stick_angle_min = stick
        stick_angle_max = stick
        arm_angle_min = arm
        arm_angle_max = arm
        
        self._randomize_pose.add_randomization(bucket_angle_min, bucket_angle_max, ["BucketPrismatic"])
        self._randomize_pose.add_randomization(stick_angle_min, stick_angle_max, ["StickPrismatic"])
        self._randomize_pose.add_randomization(arm_angle_min, arm_angle_max, ["ArmPrismatic1"])
        self._randomize_pose.add_randomization(arm_angle_min, arm_angle_max, ["ArmPrismatic2"])
        self._randomize_pose.reconfigure()
        self._randomize_pose.randomizations = []
        
        
    
    def create_shovel(self):
        terrain_shovel = agxTerrain.Shovel(self.bucket_body, self.top_edge, self.cutting_edge, self.forward_cutting_vector)
        terrain_shovel.setVerticalBladeSoilMergeDistance(0.0)
        terrain_shovel.setNoMergeExtensionDistance(0.1)
        return terrain_shovel

    def configure_materials(self, terrain: agxTerrain.Terrain):
        self.configure_track_ground_contact_materials(terrain)

        shovel_material = self.bucket_body.getGeometries()[0].getMaterial()
        terrain_material = terrain.getMaterial(agxTerrain.Terrain.MaterialType_TERRAIN)
        shovel_terrain_contact_material = simulation().getMaterialManager().getOrCreateContactMaterial(shovel_material, terrain_material)
        shovel_terrain_contact_material.setYoungsModulus(1e8)
        shovel_terrain_contact_material.setRestitution(0.0)
        shovel_terrain_contact_material.setFrictionCoefficient(0.4)

        particle_material = terrain.getMaterial(agxTerrain.Terrain.MaterialType_PARTICLE)
        bucket_particle_cm = simulation().getMaterialManager().getOrCreateContactMaterial(particle_material, self.bucket_material)
        bucket_particle_cm.setYoungsModulus(1e9)
        bucket_particle_cm.setRestitution(0.0)
        bucket_particle_cm.setFrictionCoefficient(0.7)
        bucket_particle_cm.setRollingResistanceCoefficient(0.7)

    def heuristic_control_policy(self, t):
        if t < 2.0:
            return np.array([0.0, -1.0, -1.0, 0, 0, 0])
        elif t >= 2.0 and t < 3.0:
            return np.array([-1.0, 0, 0, 0, 0, 0])
        elif t >= 3.0 and t < 7.0:
            return np.array([0.25, 0.6, 0.7, 0, 0, 0])
        elif t >= 7.0 and t < 10.0:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        elif t >= 10.0 and t < 13.5:
            return np.array([0.1, -0.8, -0.8, 0.0, 0.0, 0.0])
        return np.array([-0.0, 0, 0, 0, 0, 0])


############################################################################################################
class RandomizePose():
    def __init__(self, collection, ref_body):
        super().__init__()
        self.collection = collection
        self.ref_body = ref_body
        self.reconfigure_request = agxUtil.ReconfigureRequest()
        self.randomizations = []
    
    def add_randomization(self, min_val, max_val, joint_names):
        self.randomizations.append({"min": min_val, "max": max_val, "joint_names": joint_names})
        
    def reconfigure(self):
        cpv = agxUtil.ConstraintPositionVector()
        for item in self.randomizations:
            value = random.uniform(item['min'], item['max'])
            for name in item['joint_names']:
                joint = self.collection.getConstraint(name)
                if joint:
                    cpv.append(agxUtil.ConstraintPosition(joint, value))
                else:
                    print(f"Warning: Could not locate joint {name}")

        results = agxUtil.BodyTransformVector()
        status = self.reconfigure_request.computeTransforms(self.collection, self.ref_body, cpv, results)

        if not status:
            print("Error:", self.reconfigure_request.getErrorMessage())
        else:
            self.reconfigure_request.applyTransforms(self.collection, cpv, results, True)

############################################################################################################