import math
from typing import Callable, Tuple, Optional, Any, Dict
import numpy as np

import gymnasium.spaces as spaces
from gymnasium.envs.registration import EnvSpec
from gymnasium.core import ObsType

from agxPythonModules.utils.environment import simulation
from agxPythonModules.sensors.camera_sensors import VirtualCameraSensor
from agxPythonModules.tools.simulation_content import SimulationContent

from agxPythonModules.agxGym.agx_env import AGXGymEnv
from agxPythonModules.agxGym.utils import DisplayVirtualCameraGUI

# AGX Dynamics imports
import agx
import agxOSG
import agxSDK
import agxRender
import agxCollide
import agxTerrain

from .models.terrains import TerrainModel, LargePile, GravelPile
from .models.wheel_loader_agents import WheelLoaderAgent, WheelLoaderWA475Agent
from .models.contact_sensor import ContactSensor
from .models.rock_pile_utils import RockSpawner
from .models.bed_truck import BedTruck


class WheelLoaderTerrainEnv(AGXGymEnv):

    metadata = {"render_modes": ["human"]}
    render_mode = None
    spec: EnvSpec = EnvSpec(
        id="agx-wa475-terrain-v0",
        entry_point=None,
        max_episode_steps=1200,
        kwargs={
            "wheel_loader_agent_class": WheelLoaderWA475Agent,
            "wheel_loader_agent_kwargs": {
                "measure_energy": True
            },
            "terrain_model": GravelPile,
            "include_rigid_rocks": False,
            "include_dump_terrain": True,
            "terrain_model_kwargs": {
                "terrain_size": 30.0,
                "element_size": 0.22,
                "pile_height": 1.4
            },
            "loader_start_position": (-6.5, 0.0, 0.0),
            "loader_start_direction": (1.0, 0.0, 0.0),
            "r_coef_dump_volume": 100.0,
            "r_coef_energy": -5e-6}
    )

    # These are set after the wheel loader model is loaded
    action_space = None
    observation_space = None

    def __init__(
            self,
            wheel_loader_agent_class: Callable = WheelLoaderWA475Agent,
            wheel_loader_agent_kwargs: dict = {},
            loader_start_position: Tuple[float, float, float] = (-5.0, 0.0, 0.0),
            loader_start_direction: Tuple[float, float, float] = (1.0, 0.0, 0.0),
            terrain_model: Callable = LargePile,
            terrain_model_kwargs: dict = {},
            include_dump_terrain: bool = True,
            include_rigid_rocks: bool = False,
            rock_spawner_kwargs: dict = {},
            r_coef_load_fraction: float = 1.0,
            r_coef_mass_in_bucket: float = 1.0,
            r_coef_energy: float = 0,
            r_coef_dump_volume: float = 1.0,
            r_coef_traction: float = 0.0,
            r_coef_collision: float = 0.0,
            **kwargs):

        self._wheel_loader_model_class = wheel_loader_agent_class
        self._wheel_loader_model_kwargs = wheel_loader_agent_kwargs
        self._terrain = terrain_model
        self._terrain_model_kwargs = terrain_model_kwargs
        self._rock_spawner_kwargs = rock_spawner_kwargs

        self._loader_start_position = agx.Vec3(*loader_start_position)
        self._loader_start_direction = agx.Vec3(*loader_start_direction)

        self._include_dump_terrain = include_dump_terrain
        self._include_rigid_rocks = include_rigid_rocks
        self._r_coef_load_fraction = r_coef_load_fraction
        self._r_coef_mass_in_bucket = r_coef_mass_in_bucket
        self._r_coef_energy = r_coef_energy
        self._r_coef_dump_volume = r_coef_dump_volume
        self._r_coef_traction = r_coef_traction
        self._r_coef_collision = r_coef_collision
        self._rock_filenames = [
            "models/rock_lp_01.obj",
            "models/rock_lp_02.obj",
            "models/rock_lp_03.obj"
        ]
        super().__init__(**kwargs)
        self.observation_space, self.action_space = self._setup_gym_environment_spaces()
        heights = self._terrain.observe(o_width=7, o_length=8)
        base_area = self._terrain.element_size**2
        self._init_pile_vol = np.sum(heights * base_area)
        self._last_dump_vol = 0

    def _build_scene(self):
        sim = self.sim

        # Create terrain
        terrain: TerrainModel
        terrain = self._terrain(**self._terrain_model_kwargs)

        # Setup the Dump Truck with a separate terrain inside container
        if self._include_dump_terrain:
            bed_truck = BedTruck()
            bed_truck.setPosition(-7, -7, -0.0)
            self._bed_truck = bed_truck
            simulation().add(bed_truck)
            simulation().add(bed_truck.dump_terrain)

        def create_wall(length, pos, rot, material):
            wall = agxCollide.Geometry(agxCollide.Box(length / 2, 0.1, 1.5))
            wall.setRotation(rot)
            wall.setPosition(pos)
            wall.setMaterial(material)
            return wall

        self._side_walls = [
            create_wall(terrain.getSize().x(), agx.Vec3(0, terrain.getSize().x() / 2, 1.5),
                        agx.Quat(), terrain.getMaterial(agxTerrain.Terrain.MaterialType_TERRAIN)),
            create_wall(terrain.getSize().x(), agx.Vec3(0, -terrain.getSize().x() / 2, 1.5),
                        agx.Quat(), terrain.getMaterial(agxTerrain.Terrain.MaterialType_TERRAIN)),
            create_wall(terrain.getSize().x(), agx.Vec3(terrain.getSize().x() / 2, 0, 1.5),
                        agx.Quat(math.pi / 2, agx.Vec3().Z_AXIS()), terrain.getMaterial(agxTerrain.Terrain.MaterialType_TERRAIN)),
            create_wall(terrain.getSize().x(), agx.Vec3(-terrain.getSize().x() / 2, 0, 1.5),
                        agx.Quat(math.pi / 2, agx.Vec3().Z_AXIS()), terrain.getMaterial(agxTerrain.Terrain.MaterialType_TERRAIN))
        ]
        for w in self._side_walls:
            sim.add(w)

        # Create wheel loader
        loader: WheelLoaderAgent
        loader = self._wheel_loader_model_class(**self._wheel_loader_model_kwargs)
        self._shovel = loader.create_shovel()

        sim.add(loader)
        sim.add(terrain)
        terrain.add(self._shovel)

        loader.configure_materials(terrain)

        # Create rock spawner if rigid rocks should be included
        self._rock_spawner = None
        rock_material = agx.Material("RockMaterial")
        if self._include_rigid_rocks:
            rock_material.getBulkMaterial().setDensity(2800)
            self._rock_spawner = RockSpawner(
                self._rock_filenames,
                rock_material,
                [0.7, 0.9, 1.0],
                [1.0, 1.0, 1.0],
                terrain.frame,
                **self._rock_spawner_kwargs)
            self._rock_spawner.configure_materials(loader.bucket_body.getGeometries()[0].getMaterial(), terrain)
            sim.add(self._rock_spawner)
            sim.getSpace().setEnableCollisions(terrain.getGeometry(), self._rock_spawner.getGeometry(), False)
            for w in self._rock_spawner.emitter_walls:
                sim.getSpace().setEnableCollisions(terrain.getGeometry(), w, False)
            simulation().getMergeSplitHandler().setEnable(True)
            prop = agxSDK.MergeSplitHandler.getOrCreateProperties(terrain.getGeometry())
            prop.setEnableMergeSplit(True)
            # Create the bucket sensor.
            # The collider triggers will be updated each reset.
            # Since new rocks are created each reset.
            self._bucket_sensor = ContactSensor([], [])
            self._bed_sensor = ContactSensor([], [])

        # create contact sensors
        # Only create these if they will be used. Adding them adds 1ms of computation every timestep.
        if self._r_coef_traction > 0:
            self.tire_terrain_contact_sensors = [ContactSensor([t.getTireRigidBody()], [terrain.getGeometry()]) for t in loader.tires]
        if self._r_coef_collision > 0:
            colliding_bodies = []
            colliding_bodies.extend(self._side_walls)
            if self._include_dump_terrain:
                colliding_bodies.extend(bed_truck.colliding_bodies)
            self.loader_truck_contact_sensor = ContactSensor(loader.colliding_bodies, bed_truck.colliding_bodies + self._side_walls)

        sim.setTimeStep(1 / 60)
        sim.getSolver().setUseParallelPgs(True)
        sim.getSolver().setUse32bitGranularBodySolver(True)
        sim.getSolver().setUseGranularWarmStarting(True)
        sim.getSolver().setNumPPGSRestingIterations(10)

        #
        # For debug printing simulation content like bodies, constraints and contact materials
        #
        content = SimulationContent(simulation=simulation())
        if not content.initialize():
            print("Could not initialize simulation content viewer")
        else:
            content.print()
            cm_pp = simulation().getMaterialManager().getContactMaterial(
                terrain.getMaterial(agxTerrain.Terrain.MaterialType_PARTICLE),
                terrain.getMaterial(agxTerrain.Terrain.MaterialType_PARTICLE)
            )
            cm_sp = simulation().getMaterialManager().getContactMaterial(
                content.bodies["Bucket"].getGeometries()[0].getMaterial(),
                terrain.getMaterial(agxTerrain.Terrain.MaterialType_PARTICLE)
            )
            cm_st = simulation().getMaterialManager().getContactMaterial(
                content.bodies["Bucket"].getGeometries()[0].getMaterial(),
                terrain.getMaterial(agxTerrain.Terrain.MaterialType_TERRAIN)
            )
            print("Additional contact materials")
            content.printContactMaterial(cm_pp)
            content.printContactMaterial(cm_sp)
            content.printContactMaterial(cm_st)
            content.printStats()

        self._loader = loader
        self._terrain = terrain
        self._terrain_renderers = []

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsType, Dict[str, Any]]:
        super(AGXGymEnv, self).reset(seed=seed)
        # set loader position in x / y and forward direction
        self._loader.reset(self._loader_start_position, self._loader_start_direction)
        self._loader.set_action(np.asarray([0] * self._loader.action_range()[0].shape[0]))

        self._terrain.reset(self.np_random)
        if self._include_dump_terrain:
            self._bed_truck.dump_terrain.reset()
        for r in self._terrain_renderers:
            r.last(self.sim.getTimeStamp())

        if len(self._loader.tires) > 0:
            point, t = self._terrain.get_highest_point_under_tires(self._loader.tires)
            self._loader.set_position_relative_body(t.getTireRigidBody(), point + t.getRadius())

        if self._include_rigid_rocks:
            self._bucket_sensor.clear()
            self._bed_sensor.clear()
            self._rock_spawner.reset(self.np_random)
            self._bucket_sensor.add_colliders([self._loader.bucket_sensor_geom], self._rock_spawner.current_cm_colliders)
            self._bed_sensor.add_colliders([self._bed_truck.bed_sensor_geom], self._rock_spawner.current_cm_colliders)

        self.episode_step = 0
        self.sim.setTimeStamp(0.0)
        self._last_dump_vol = 0
        heights = self._terrain.observe(o_width=7, o_length=8)
        base_area = self._terrain.element_size**2
        self._init_pile_vol = np.sum(heights * base_area)
        o, _, _, _, d = self._observe()
        return o, d

    def _modify_visuals(self, root):
        # Setup a renderer for the terrain. Here we choose to only render the height field but with height coloring
        renderer = self._terrain.create_or_get_renderer(root)
        self.sim.add(renderer)
        self._terrain_renderers.append(renderer)

        if self._include_dump_terrain:
            node = agxOSG.findGeometryNode(self._bed_truck.dump_terrain.getGeometry(), root)
            agxOSG.setAlpha(node, 0.0)
            renderer2 = self._bed_truck.dump_terrain.create_or_get_renderer(root)
            self.sim.add(renderer2)
            self._terrain_renderers.append(renderer2)

        self.app.getSceneDecorator().setBackgroundColor(agxRender.Color.BlanchedAlmond(), agxRender.Color.DimGray())

        camera_data = self.app.getCameraData()
        camera_data.nearClippingPlane = 0.1
        camera_data.farClippingPlane = 100
        self.app.applyCameraData(camera_data)
        eye = agx.Vec3(1.8279971900714806E+00, -3.0288331910504173E+01, 2.0395564680074859E+01)
        center = agx.Vec3(1.7803797752229009E+00, -2.9435514056186478E+00, 7.8468583291171912E-01)
        up = agx.Vec3(-0.0340, 0.5824, 0.8122)
        self.app.setCameraHome(eye, center, up)

        self.app.getSceneDecorator().setEnableLogo(False)

        if self._include_rigid_rocks:
            agxOSG.createVisual(self._rock_spawner, root)
            # Uncomment to visualize the rock emitter
            # agxOSG.createVisual(self._rock_spawner._emitter_geom, root)
            # for w in self._rock_spawner._emitter_walls:
            #     agxOSG.createVisual(w, root)

        for w in self._side_walls:
            agxOSG.createVisual(w, root)

    def _setup_virtual_cameras(self, visual_observation_space: bool = False):
        # Creates one virtual camera sensor looking at the cart.
        # This can be used as the only observation if the observation space is changed with a wrapper.
        # That is demonstrated below.
        camera = VirtualCameraSensor(
            128,
            128,
            self._loader.bucket_body.getPosition() + agx.Vec3(0, 0, 0.6),
            self._loader.bucket_body.getPosition() + agx.Vec3(3, 0, -0.6),
            agx.Vec3().Z_AXIS(),
            fovy=78,
            near=0.1,
            far=100.0,
            depth_camera=False)
        self.virtual_cameras.append(camera)

        # TODO: setup a dict observation space with both scalars and cameras
        if visual_observation_space:
            pass

        # This is not headless. So I want a gui for viewing.
        if not self.headless:
            self.virtual_camera_gui = DisplayVirtualCameraGUI(self.virtual_cameras, 512, 512)

    def _setup_gym_environment_spaces(self):
        o_loader = self._loader.observe()
        o_low, o_high = self._loader.observation_range()
        a_low, a_high = self._loader.action_range()

        observation_space = spaces.Box(
            low=o_low,
            high=o_high,
            shape=(o_loader.shape[0],),
            dtype=np.float64
        )
        action_space = spaces.Box(low=a_low, high=a_high, shape=(a_low.shape[0],), dtype=np.float64)
        return observation_space, action_space

    def _observe(self):
        # If we have a camera it is used as an observation and we
        # want to move it relative the cart body
        if len(self.virtual_cameras) > 0:
            self.virtual_cameras[0].transform_relative_body(
                self._loader.virtual_camera_ref_body,
                self._loader.virtual_camera_eye,
                self._loader.virtual_camera_center,
                self._loader.virtual_camera_up)
        o = self._loader.observe()

        truncated = False
        if self.spec.max_episode_steps is not None:
            truncated = self.episode_step >= self.spec.max_episode_steps
        r = self._reward()

        return o, r, False, truncated, {}

    def _set_action(self, action):
        # If keyboard control has been called we ignore any action here
        # since they alreay have been set by the event listeners
        if self._loader.keyboard_controls is not None:
            return
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._loader.set_action(action)

    def _pile_removed_volume(self):
        ''' How much of the pile has been removed '''
        heights = self._terrain.observe(o_width=7, o_length=8)
        base_area = self._terrain._element_size**2
        return max(0, self._init_pile_vol - np.sum(heights * base_area))

    def _reward_dump_volume(self):
        ''' How much volume has been added to the dump pile since last step '''
        if not self._include_dump_terrain:
            return 0
        heights = self._bed_truck.dump_terrain.observe()
        base_area = self._bed_truck.dump_terrain._element_size**2
        dump_vol = min(np.sum(heights * base_area), self._pile_removed_volume())

        vol_change = dump_vol - self._last_dump_vol
        self._last_dump_vol = dump_vol
        return vol_change

    def _reward_dump_rock_mass(self):
        ''' How much rock mass has been added to the dump pile '''
        rock_mass = 0
        if self._include_rigid_rocks:
            for rb in self._rock_spawner.current_emitted_bodies:
                if self._bed_sensor.in_contact_pairs(rb):
                    rock_mass += rb.getMassProperties().getMass()
        return rock_mass

    def _reward_rock_mass_in_pile_dump_and_bucket(self) -> Tuple[float, float, float]:
        '''
        Returns a tuple with the mass of the rocks in the pile, the dump truck and the bucket
        '''
        dump_rock_mass = 0
        bucket_rock_mass = 0
        pile_rock_mass = 0
        if self._include_rigid_rocks:
            for rb in self._rock_spawner.current_emitted_bodies:
                if self._bed_sensor.in_contact_pairs(rb):
                    dump_rock_mass += rb.getMassProperties().getMass()
                    continue
                if self._bucket_sensor.in_contact_pairs(rb):
                    bucket_rock_mass += rb.getMassProperties().getMass()
                    continue
                pile_rock_mass += rb.getMassProperties().getMass()
        return (pile_rock_mass, dump_rock_mass, bucket_rock_mass)

    def _reward_terrain_mass_in_bucket(self):
        ''' How terrain mass is there in the bucket '''
        return self._terrain.getDynamicMass(self._shovel)

    def _reward_energy_consumption(self):
        ''' How power is the wheel laoder using in this instance '''
        return self._loader.energy_consumption

    def _reward_load_fraction(self):
        '''
        The fraction of the volume of the shovel that is filled
        '''
        return self._terrain.getLastDeadLoadFraction(self._shovel)

    def _reward_total_mass_in_bucket(self):
        '''
        Dynamic mass in given shovel, including both particles and fluid mass.
        Pluss the mass of Rigid rocks in the shovel
        '''
        terrain_mass = self._terrain.getDynamicMass(self._shovel)
        rock_mass = 0
        if self._include_rigid_rocks:
            for rb in self._rock_spawner.current_emitted_bodies:
                if self._bed_sensor.in_contact_pairs(rb):
                    rock_mass += rb.getMassProperties().getMass()
        return terrain_mass + rock_mass

    def _reward_traction_on_tires(self):
        '''
        returns 1 if all four tires has traction on the terrain. 0 otherwise
        '''
        for sensor in self.tire_terrain_contact_sensors:
            if not sensor.contact:
                return 0
        return 1

    def _reward_collision(self):
        '''
        Returns 1 if the wheel loader has not collided with anything. 0 otherwise.
        '''
        if not self.loader_truck_contact_sensor.contact:
            return 1
        return 0

    def _reward_nb_rocks_in_bucket(self):
        if not self._include_rigid_rocks:
            return 0
        return self._bucket_sensor.nb_contacts

    def _reward(self):
        '''
        Change the method to define the task that you want the wheel loader agent to solve.

        Right now it wants to learn to move material from the pile to the dump volume using as little energy as possible
        '''
        return self._r_coef_dump_volume * self._reward_dump_volume() + self._r_coef_energy * self._reward_energy_consumption()

    def _terminal(self):
        return self.episode_step > self.spec.max_episode_steps

    def heuristic_control_policy(self, t):
        return self._loader.heuristic_control_policy(t)

    def keyboard_control_policy(self, t):
        # enable keybord controls if not already set
        if self._loader.keyboard_controls is None:
            print("Enabling keyboard controls")
            self._loader.keyboard_controls = self._wheel_loader_model_class.default_keyboard_controls()
