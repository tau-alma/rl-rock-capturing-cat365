import agx
import agxOSG
import agxSDK
import agxUtil
import agxTerrain
import agxCollide

import os
import math
import logging
from typing import AnyStr, List, Tuple

from agxPythonModules.tools.read_file import SimulationFile
from agxPythonModules.utils.environment import root
from agxPythonModules.utils.environment import simulation

import numpy as np

logger = logging.getLogger(__name__)


def inertia_tensor_size_estimation(mp):
    ''' Returns the an approximation of a rock size given its diagonalized inertia tensor '''
    m = mp.getMass()
    it = mp.getInertiaTensor()
    np_it = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            np_it[i, j] = it.at(i, j)

    (w, v) = np.linalg.eig(np_it)

    Ixx = w[0]
    Iyy = w[1]
    Izz = w[2]
    x2 = 6 / m * (Izz + Iyy - Ixx)
    y2 = 6 / m * (Izz + Ixx - Iyy)
    z2 = 6 / m * (Iyy + Ixx - Izz)

    # Find the middle coordinate
    size = 0
    if x2 > y2:
        if x2 < z2:
            size = math.sqrt(x2)
        elif z2 < y2:
            size = math.sqrt(y2)
        else:
            size = math.sqrt(z2)
    else:
        if y2 < z2:
            size = math.sqrt(y2)
        elif z2 < x2:
            size = math.sqrt(x2)
        else:
            size = math.sqrt(z2)
    return size


def create_rigid_body_from_obj(name, transformation, material):
    convex_shapes = agxCollide.ConvexRefVector()
    agxUtil.createConvexDecomposition(name, convex_shapes, 50, transformation)
    rb = agx.RigidBody()
    for c in convex_shapes:
        assert c
        g = agxCollide.Geometry(c.asConvex())
        if material:
            g.setMaterial(material)
        rb.add(g)
    cm_sphere = agxCollide.Geometry(agxCollide.Sphere(0.01))
    cm_sphere.setName("cm_sphere")
    cm_sphere.setPosition(rb.getCmLocalTranslate())
    rb.add(cm_sphere)
    return rb


def create_distribution_table(shape_paths, sizes, weights, material):
    dt = agx.EmitterDistributionTable()
    n = len(shape_paths) * len(sizes)

    for shape_path in shape_paths:
        # Get the original size of the model
        rb_original = create_rigid_body_from_obj(shape_path, agx.Matrix3x3(), None)
        size_original = inertia_tensor_size_estimation(rb_original.getMassProperties())
        # add a correctly transformed rb to the distribution table for each size
        for s, w in zip(sizes, weights):
            transformation = agx.Matrix3x3() * s / size_original
            rb = create_rigid_body_from_obj(shape_path, transformation, material)
            prop = agxSDK.MergeSplitHandler.getOrCreateProperties(rb)
            prop.setEnableMergeSplit(True)
            model = agx.RigidBodyEmitterDistributionModel(rb, w / n)
            dt.addModel(model)
    return dt


def create_rigid_body_emitter(emitter_geom,
                              rock_file_names,
                              sizes,
                              weights,
                              rock_material):
    emitter = agx.RigidBodyEmitter()
    emitter.setQuantity(agx.RigidBodyEmitter.QUANTITY_COUNT)

    dt = create_distribution_table(rock_file_names, sizes, weights, rock_material)
    emitter.setDistributionTable(dt)

    emitter.setGeometry(emitter_geom)

    return emitter


class RockSpawner(agx.RigidBodyEmitter):
    def __init__(self,
                 rock_file_names,
                 rock_material: agx.Material,
                 sizes: List[float],
                 weights: List[float],
                 parent_frame: agx.Frame,
                 max_nb_rocks: int = 4,
                 position_offset: Tuple[float, float, float] = (0, 0, 0),
                 emitter_size: Tuple[float, float, float] = (1, 1, 1.5),
                 reset_timer: float = 4.0
                 ):
        super().__init__()
        self._reset_timer = reset_timer
        self._emitter_size = agx.Vec3(*emitter_size)

        self.setQuantity(agx.RigidBodyEmitter.QUANTITY_COUNT)
        dt = create_distribution_table(rock_file_names, sizes, weights, rock_material)
        self.setDistributionTable(dt)

        emitter_geom = agxCollide.Geometry(agxCollide.Box(self._emitter_size))
        emitter_geom.setParentFrame(parent_frame)
        emitter_geom.setLocalPosition(agx.Vec3(*position_offset) + agx.Vec3(0, 0, self._emitter_size.z()))
        emitter_geom.setSensor(True)
        emitter_geom.setEnableCollisions(False)
        simulation().add(emitter_geom)

        emitter_wall0 = agxCollide.Geometry(agxCollide.Box(0.1, 1.2 * self._emitter_size.y(), 2 * self._emitter_size.z()))
        emitter_wall0.setParentFrame(emitter_geom.getFrame())
        emitter_wall0.setLocalPosition(agx.Vec3(-self._emitter_size.x(), 0, self._emitter_size.z()))
        emitter_wall0.setEnableCollisions(False)
        simulation().add(emitter_wall0)
        emitter_wall1 = agxCollide.Geometry(agxCollide.Box(0.1, 1.2 * self._emitter_size.y(), 2 * self._emitter_size.z()))
        emitter_wall1.setParentFrame(emitter_geom.getFrame())
        emitter_wall1.setLocalPosition(agx.Vec3(0, self._emitter_size.y() * 1.2, self._emitter_size.z()))
        emitter_wall1.setRotation(agx.Quat(math.pi / 2, agx.Vec3().Z_AXIS()))
        emitter_wall1.setEnableCollisions(False)
        simulation().add(emitter_wall1)
        emitter_wall2 = agxCollide.Geometry(agxCollide.Box(0.1, 1.2 * self._emitter_size.y(), 2 * self._emitter_size.z()))
        emitter_wall2.setParentFrame(emitter_geom.getFrame())
        emitter_wall2.setLocalPosition(agx.Vec3(0, -self._emitter_size.y() * 1.2, self._emitter_size.z()))
        emitter_wall2.setRotation(agx.Quat(math.pi / 2, agx.Vec3().Z_AXIS()))
        emitter_wall2.setEnableCollisions(False)
        simulation().add(emitter_wall2)

        self.setGeometry(emitter_geom)
        self.setMaximumEmittedQuantity(2)
        self.setRate(self.getMaximumEmittedQuantity() / simulation().getTimeStep())

        self._current_emitted_bodies = []
        self._emitter_walls = [emitter_wall0, emitter_wall1, emitter_wall2]
        self._emitter_geom = emitter_geom
        self._rock_material = rock_material
        self._max_nb_rocks = max_nb_rocks

    def reset(self, np_rng: np.random.RandomState):
        for w in self._emitter_walls:
            w.setEnableCollisions(True)

        self.remove_emitted_bodies()
        self.setMaximumEmittedQuantity(self.getEmittedQuantity() + np_rng.choice(np.arange(1, self._max_nb_rocks)))
        simulation().stepTo(simulation().getTimeStamp() + self._reset_timer)
        for w in self._emitter_walls:
            w.setEnableCollisions(False)

        for rb in self.getEmittedBodiesInSimulation(simulation()):
            self._current_emitted_bodies.append(rb)

            if root() is not None:
                for g in rb.getGeometries():
                    agxOSG.setTexture(g, root(), "stone-texture09.jpg")

    def configure_materials(self, shovel_material: agx.Material, terrain: agxTerrain.Terrain):
        rock_rock_cm = agx.ContactMaterial(self._rock_material, self._rock_material)
        rock_rock_cm.setYoungsModulus(1e8)
        rock_rock_cm.setRestitution(0.0)
        rock_rock_cm.setFrictionModel(agx.BoxFrictionModel(agx.FrictionModel.DIRECT))
        rock_rock_cm.setFrictionCoefficient(0.6, agx.ContactMaterial.PRIMARY_DIRECTION)
        rock_rock_cm.setFrictionCoefficient(0.6, agx.ContactMaterial.SECONDARY_DIRECTION)
        simulation().add(rock_rock_cm)
        shovel_rock_cm = agx.ContactMaterial(shovel_material, self._rock_material)
        shovel_rock_cm.setYoungsModulus(1e8)
        shovel_rock_cm.setRestitution(0.0)
        shovel_rock_cm.setFrictionModel(agx.BoxFrictionModel(agx.FrictionModel.DIRECT))
        shovel_rock_cm.setFrictionCoefficient(0.6, agx.ContactMaterial.PRIMARY_DIRECTION)
        shovel_rock_cm.setFrictionCoefficient(0.6, agx.ContactMaterial.SECONDARY_DIRECTION)
        simulation().add(shovel_rock_cm)
        rock_terrain_cm = agx.ContactMaterial(terrain.getMaterial(agxTerrain.Terrain.MaterialType_TERRAIN), self._rock_material)
        rock_terrain_cm.setYoungsModulus(1e8)
        rock_terrain_cm.setRestitution(0.0)
        rock_terrain_cm.setFrictionCoefficient(0.6, agx.ContactMaterial.PRIMARY_DIRECTION)
        rock_terrain_cm.setFrictionCoefficient(0.6, agx.ContactMaterial.SECONDARY_DIRECTION)
        simulation().add(rock_terrain_cm)
        rock_particle_cm = agx.ContactMaterial(terrain.getMaterial(agxTerrain.Terrain.MaterialType_PARTICLE), self._rock_material)
        rock_particle_cm.setYoungsModulus(1e8)
        rock_particle_cm.setRestitution(0.0)
        rock_particle_cm.setFrictionCoefficient(0.6, agx.ContactMaterial.PRIMARY_DIRECTION)
        rock_particle_cm.setFrictionCoefficient(0.6, agx.ContactMaterial.SECONDARY_DIRECTION)
        simulation().add(rock_particle_cm)

    def remove_emitted_bodies(self):
        for rb in self._current_emitted_bodies:
            simulation().remove(rb)
        self._current_emitted_bodies = []

    @property
    def emitter_walls(self) -> List[agxCollide.Geometry]:
        return self._emitter_walls

    @property
    def current_emitted_bodies(self) -> List[agx.RigidBody]:
        return self._current_emitted_bodies

    @property
    def current_cm_colliders(self) -> List[agxCollide.Geometry]:
        cm_colliders = []
        for rb in self.current_emitted_bodies:
            for g in rb.getGeometries():
                if "cm_sphere" in g.getName():
                    cm_colliders.append(g)
        return cm_colliders


class RockPileModel(agxSDK.Assembly):
    def __init__(self, filename: AnyStr):
        super(RockPileModel, self).__init__()
        self._path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

        temp_simulation = agxSDK.Simulation()
        data = SimulationFile(simulation=temp_simulation, root=root())
        if not data.load(self._path, parent=self):
            raise FileNotFoundError(f"Unable to load rock pile model: {self._path}")

        states = {}
        for rb in self.getRigidBodies():
            states[rb.getUuid()] = rb.getEnable()

        self._time_step = temp_simulation.getTimeStep()
        self._num_ri = temp_simulation.getSolver().getNumRestingIterations()

        self._initialize(data)

        del data
        del temp_simulation

        for rb in self.getRigidBodies():
            if not rb.getUuid() in states:
                continue
            rb.setEnable(states[rb.getUuid()])

    def _initialize(self, data: SimulationFile):
        # get mold body
        self._mold_body = None
        self._rock_body = None

        for k, rb in data.bodies.items():
            if rb.getMotionControl() == agx.RigidBody.STATIC:
                self._mold_body = rb
            elif rb.getMotionControl() == agx.RigidBody.DYNAMICS and "emitted" in rb.getName():
                self._rock_body = rb
            # make sure that amor is enabled
            prop = agxSDK.MergeSplitHandler.getOrCreateProperties(rb)
            prop.setEnableMergeSplit(True)

        # get material for mold and rock body
        self._mold_material = self._mold_body.getGeometries()[0].getMaterial()
        self._rock_material = self._rock_body.getGeometries()[0].getMaterial()

    def create_contact_material(self, manager: agxSDK.MaterialManager, material: agx.Material):
        fm = agx.IterativeProjectedConeFriction()
        fm.setSolveType(agx.FrictionModel.ITERATIVE)

        cm1: agx.ContactMaterial = manager.getOrCreateContactMaterial(material, self._rock_material)
        cm1.setFrictionCoefficient(0.4)
        cm1.setRestitution(0.4)
        cm1.setYoungsModulus(1e9)
        cm1.setFrictionModel(fm)

        cm2: agx.ContactMaterial = manager.getOrCreateContactMaterial(material, self._mold_material)
        cm2.setFrictionCoefficient(0.4)
        cm2.setRestitution(0.4)
        cm2.setYoungsModulus(1e9)
        cm2.setFrictionModel(fm)

        return cm1, cm2
