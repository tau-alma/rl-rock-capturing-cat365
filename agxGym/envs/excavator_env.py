import math
from typing import Callable, Tuple, Optional, Any, Dict
import numpy as np
import random

import gymnasium.spaces as spaces
from gymnasium.envs.registration import EnvSpec
from gymnasium.core import ObsType

# AGX Dynamics imports
import agx
import agxRender
import agxCollide
import agxTerrain
import agxSDK
import agxIO
import agxOSG
import agxModel

from agxPythonModules.utils.environment import simulation
from agxPythonModules.tools.simulation_content import SimulationContent

from agxPythonModules.agxGym.agx_env import AGXGymEnv

from .models.terrains import FlatTerrain, TerrainModel
from .models.excavator_365_agent import Excavator365Agent


class ExcavatorTerrainEnv(AGXGymEnv):

    metadata = {"render_modes": ["human"]}
    render_mode = None
    spec: EnvSpec = EnvSpec(
        id="agx-365-terrain-rock-v0",
        entry_point=None,
        max_episode_steps=500, # 500 - 360 is ok
        kwargs={
            "terrain_model_kwargs": {
                "terrain_size_x": 35.0,
                "terrain_size_y": 35.0,
                "element_size": 0.22,
                "terrain_material": "DIRT_1"
            }})

    # These are set after the excavator model is loaded
    action_space = None
    observation_space = None

    def __init__(
            self,
            excavator_start_position: Tuple[float, float, float] = (1.0, 0.0, 0.0),
            excavator_start_direction: Tuple[float, float, float] = (1.0, 0.0, 0.0),
            rock_start_position: Tuple[float, float, float] = (-10.0, 0.5, 0.5), 
            rock_start_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0),
            terrain_model: Callable = FlatTerrain,
            terrain_model_kwargs: dict = {},
            r_ditch: float = 0.0,
            r_coef_energy: float = 0,
            **kwargs):

        self._excavator_start_position = agx.Vec3(*excavator_start_position)
        self._loader_start_direction = agx.Vec3(*excavator_start_direction)
        self._rock_start_position = agx.Vec3(*rock_start_position)
        self._rock_start_velocity = agx.Vec3(*rock_start_velocity)
        self._terrain_model = terrain_model
        self._terrain_model_kwargs = terrain_model_kwargs
        
        self._random_target_position = True
        if self._random_target_position:
            self._target_position  = self._generate_random_target_position()
        else:
            self._target_position = np.array([-6.0, 0.0, 2.0])
            
            
        self._current_action = None
        self._prev_action = None
        self._smoothing_penalty = None
        
        self._random_density_rock = True
        self._initial_random_rotation_rock = False
        self._random_geometry_rock = True
        
        self._initial_random_position_rock = True
        self._initial_excavator_configuration = True
        
        super().__init__(**kwargs)
        self.observation_space, self.action_space = self._setup_gym_environment_spaces()

    def _build_scene(self):
        sim = self.sim

        # Create terrain
        terrain: TerrainModel
        terrain = self._terrain_model(**self._terrain_model_kwargs)
        sim.add(terrain)

        terrain.getTerrainMaterial().getCompactionProperties().setAngleOfReposeCompactionRate(24.0)
        terrain.getTerrainMaterial().getBulkProperties().setYoungsModulus(1e6)
        ############################################################################################################
        # Add rock
        rock_file = "models/convex_stones/convex_rock2.obj"
        # rock_file = "models/convex_stones/convex_rock3.obj"
        # rock_file = "models/convex_stones/convex_rock4.obj"
        # rock_file = "models/convex_stones/convex_rock5.obj"
        # rock_file = "models/convex_stones/convex_rock6.obj"
        
        rock_material = agx.Material("Rocks")
        
        if self._random_density_rock:
            density = self._generate_random_rock_density()
        else:
            density = 2000.0
        rock_material.getBulkMaterial().setDensity(density)
        
        # rock_material.getBulkMaterial().setDensity(2000)
        
        mesh_reader = agxIO.MeshReader()
        mesh_reader.readFile(rock_file)
        
        scale = agx.Vec3(0.001) 
        
        vertices = mesh_reader.getVertices()
        scaled_vertices = agx.Vec3Vector()
        for v in vertices:
            scaled_vertices.append(agx.Vec3.mul(v, scale))

        # scaled_vertices = scale_mesh(mesh_reader.getVertices(), agx.Vec3(0.001)) # 0.0009
        mesh = agxCollide.Convex(scaled_vertices, mesh_reader.getIndices(), "rock")
        
        rock_geometry = agxCollide.Geometry(mesh.shallowCopy())
        rock_geometry.setMaterial(rock_material)
        rock = agx.RigidBody(rock_geometry)
        ################################
        if self._initial_random_rotation_rock:
            random_vector = agx.Vec3(random.random(), random.random(), random.random())
            random_angle = random.uniform(0, 2 * math.pi)
        else:
            random_vector = agx.Vec3(0, 0, 0)
            random_angle = 0
        
        if self._initial_random_position_rock:
            self._rock_start_position = self._generate_random_rock_position()
        
        rock.setRotation(agx.Quat(random_angle, random_vector))  
        # rock.setPosition(self._rock_start_position)
        rock.setCmPosition(self._rock_start_position)
        rock.setVelocity(self._rock_start_velocity)
        ################################
        rock.setName("Rock")
        sim.add(rock)
        self._rock = rock
        agxSDK.MergeSplitHandler.getOrCreateProperties(rock).setEnableMergeSplit(True)
        
        # Set contact materials of the terrain and rock
        terrain_material = terrain.getMaterial(agxTerrain.Terrain.MaterialType_TERRAIN)
        rock_terrain_contact_material = sim.getMaterialManager().getOrCreateContactMaterial(rock_material, terrain_material)
        sim.add(rock_terrain_contact_material)
        
        particle_material = terrain.getMaterial(agxTerrain.Terrain.MaterialType_PARTICLE)
        rock_particle_contact_material = sim.getMaterialManager().getOrCreateContactMaterial(particle_material, rock_material)
        sim.add(rock_particle_contact_material)
        
        ############################################################################################################
        excavator = Excavator365Agent()
        sim.add(excavator)
        excavator.configure_materials(terrain)
        self._shovel = excavator.create_shovel()
        terrain.add(self._shovel)

        sim.setTimeStep(1 / 60)
        sim.getSolver().setUseParallelPgs(True)
        sim.getSolver().setUse32bitGranularBodySolver(True)
        sim.getSolver().setUseGranularWarmStarting(True)
        sim.getSolver().setNumPPGSRestingIterations(10)

        ############################################################################################################
        #
        # For debug printing simulation content like bodies, constraints and contact materials
        #
        content = SimulationContent(simulation=simulation())
        if not content.initialize():
            print("Could not initialize simulation content viewer")
        else:
            # content.print()
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
            cm_rp = simulation().getMaterialManager().getContactMaterial(
                terrain.getMaterial(agxTerrain.Terrain.MaterialType_PARTICLE),
                content.bodies["Rock"].getGeometries()[0].getMaterial()
            )
            cm_rt = simulation().getMaterialManager().getContactMaterial(
                content.bodies["Rock"].getGeometries()[0].getMaterial(),
                terrain.getMaterial(agxTerrain.Terrain.MaterialType_TERRAIN)
            )
            print("Additional contact materials")
            content.printContactMaterial(cm_pp)
            content.printContactMaterial(cm_sp)
            content.printContactMaterial(cm_st)
            content.printContactMaterial(cm_rp)
            content.printContactMaterial(cm_rt)
            content.printStats()

        self._terrain = terrain
        self._excavator = excavator
        self._terrain_renderers = []

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsType, Dict[str, Any]]:
        super(AGXGymEnv, self).reset(seed=seed)
        
        if self._initial_random_rotation_rock:
            random_vector = agx.Vec3(random.random(), random.random(), random.random())
            random_angle = random.uniform(0, 2 * math.pi)
        else:
            random_vector = agx.Vec3(0, 0, 0)
            random_angle = 0
        
        if self._initial_random_position_rock:
            self._rock_start_position = self._generate_random_rock_position()
        
        if self._initial_excavator_configuration:
            excavator_config = self._generate_excavator_configuration(self._rock_start_position)
            excavator_config = agx.Vec3(excavator_config[0], excavator_config[1], excavator_config[2])
        else:
            excavator_config = agx.Vec3(-0.096, -0.6, -0.661)
            
        if self._random_geometry_rock:
            new_shape = self._generate_random_geometry_rock()
            geo = self._rock.getGeometries()[0]
            geo.replace(0, new_shape)    
        
        if self._random_density_rock:
            density = self._generate_random_rock_density()
            self._rock.getGeometries()[0].getMaterial().getBulkMaterial().setDensity(density)
            self._rock.updateMassProperties()
            # print("mass ",self._rock.getMassProperties().getMass()) 
            # print("rock density: ", self._rock.getGeometries()[0].getMaterial().getBulkMaterial().getDensity())
            
            
        self._rock.setRotation(agx.Quat(random_angle, random_vector))
        # self._rock.setPosition(self._rock_start_position)
        self._rock.setCmPosition(self._rock_start_position)
        self._rock.setVelocity(self._rock_start_velocity)
        #######################################################################
        # set loader position in x / y and forward direction
        self._excavator.reset(self._excavator_start_position, excavator_config)
        self._excavator.set_action_bucket_arm_boom(np.asarray([0] * self._excavator.action_range_bucket_arm_boom()[0].shape[0]))
        self._terrain.reset(self.np_random)
        
        for r in self._terrain_renderers:
            r.last(self.sim.getTimeStamp())
        
        #######################################################################
        if self._random_target_position:
            self._target_position  = self._generate_random_target_position()
        else:
            self._target_position = np.array([-6.0, 0.0, 2.0])      
        
        #######################################################################
        self._current_action = None
        self._prev_action = None
        self._smoothing_penalty = None
        
        self.episode_step = 0
        self.sim.setTimeStamp(0.0)
        #############################################################################
        # # for body in excavator.getRigidBodies():
        # #     print(body)
        # chain = agxModel.SerialKinematicChain(simulation(), self._excavator.under_carriage_body, self._excavator.bucket_body)
        # # print('number of joints',chain.getNumJoints())
        # # print('number of links',chain.getNumLinks())
        # # print('Is Valid?', chain.isValid())
        # # for joint in chain.getJoints():
        # #     print(joint.getName())
        
        # T_bucket_in_under_carriage_body = self._bucket_transformation_in_under_carriage_body()
        # q_current = [self._excavator.cabin_hinge.getAngle(),
        #            self._excavator.arm_prismatics[0].getAngle(),
        #            self._excavator.stick_prismatic.getAngle(),
        #            self._excavator.bucket_prismatic.getAngle()]
        
        # IKstatus, q_desired = chain.computeInverseKinematics(T_bucket_in_under_carriage_body, q_current)
        # if IKstatus!=0:
        #     print("Invalid IK solution, Status:", IKstatus)
        
        # FKstatus, T_bucket_in_under_carriage_body_Estimation = chain.computeForwardKinematics(q_current,clampToJointLimits=True)
        # if FKstatus!=0:
        #     print("Invalid FK solution, Status:", FKstatus)
        #############################################################################
        o, _, _, _, d = self._observe()
        return o, d

    def _modify_visuals(self, root):
        # Setup a renderer for the terrain. Here we choose to only render the height field but with height coloring
        renderer = self._terrain.create_or_get_renderer(root)
        self.sim.add(renderer)
        self._terrain_renderers.append(renderer)

        self.app.getSceneDecorator().setBackgroundColor(agxRender.Color.BlanchedAlmond(), agxRender.Color.DimGray())

        self._excavator.track(Excavator365Agent.Location.LEFT).create_shoe_visual(Excavator365Agent.default_track_settings())
        self._excavator.track(Excavator365Agent.Location.RIGHT).create_shoe_visual(Excavator365Agent.default_track_settings())

        camera_data = self.app.getCameraData()
        camera_data.nearClippingPlane = 0.1
        camera_data.farClippingPlane = 100
        self.app.applyCameraData(camera_data)
        eye = agx.Vec3(1.8279971900714806E+00, -3.0288331910504173E+01, 2.0395564680074859E+01)
        center = agx.Vec3(1.7803797752229009E+00, -2.9435514056186478E+00, 7.8468583291171912E-01)
        up = agx.Vec3(-0.0340, 0.5824, 0.8122)
        self.app.setCameraHome(eye, center, up)

        ###########################################################
        # root = self.app.getRoot()
        # self._create_visualized_objects(self.sim,root) 
        ###########################################################
        
        self.app.getSceneDecorator().setEnableLogo(False)

    def _set_action(self, action):
        # If keyboard control has been called we ignore any action here
        # since they alreay have been set by the event listeners
        if self._excavator.keyboard_controls is not None:
            return
        action = np.clip(action, self.action_space.low, self.action_space.high)
        ###################################################################
        self._current_action = action
        
        if self._prev_action is not None:
            self._smoothing_penalty = np.linalg.norm(self._current_action - self._prev_action, ord=2)
        else:
            self._smoothing_penalty = 0.0
            
        # Store the current action for next step
        self._prev_action = action
        ###################################################################
        self._excavator.set_action_bucket_arm_boom(action)
    
    def _reward_terrain_mass_in_bucket(self):
        ''' How terrain mass is there in the bucket '''
        return self._terrain.getDynamicMass(self._shovel)

    def _terminal(self):
        terminal = False 
        return terminal
    
    def _reward_control_input_without_force(self):
        r = 0
        if self._current_action is not None:
            # r = -0.1 * (1/1.74) * np.linalg.norm(self._current_action, ord=2)
            r = -1.0 * (1/3) * np.linalg.norm(self._current_action, ord=2)**2
        return r
    
    def _reward_control_input(self):
        r = 0
        arm_prismatic_force = self._excavator.arm_prismatics[0].getMotor1D().getCurrentForce()
        stick_prismatic_force = self._excavator.stick_prismatic.getMotor1D().getCurrentForce()
        bucket_prismatic_force = self._excavator.bucket_prismatic.getMotor1D().getCurrentForce()
        force = np.array([arm_prismatic_force, stick_prismatic_force, bucket_prismatic_force])
        if self._current_action is not None:
            # r = -0.1 * (1/1.74) * np.linalg.norm(self._current_action, ord=2)
            temp = np.array([arm_prismatic_force*self._current_action[0],
                             stick_prismatic_force*self._current_action[1],
                             bucket_prismatic_force*self._current_action[2]])
            r = -1.0 * (1/3) *((1/2e5)**2) * np.linalg.norm(temp, ord=2)**2
        # print("reward_control_input: ",r)
        return r
    
    def _reward_smoothing_control_input(self):
        r = 0
        if self._smoothing_penalty is not None:
            # r = -1.0 * (1/3.47) * self._smoothing_penalty
            r = -1.0 * (1/12) * self._smoothing_penalty**2
        return r
    
    
    def _setup_gym_environment_spaces(self):
        o_loader = self._excavator.observe()
        o_low, o_high = self._excavator.observation_range()
        a_low, a_high = self._excavator.action_range_bucket_arm_boom()

        ##############################################################
        # Define the low and high values for the bucket and rock position
        bucket_position_low = np.array([-12.0, -2.5], dtype=np.float64)
        bucket_position_high = np.array([-4.0, 8.5], dtype=np.float64)
        rock_position_low = np.array([-12.0, -2.5], dtype=np.float64)
        rock_position_high = np.array([-4.0, 8.5], dtype=np.float64)
        target_position_low = np.array([-12.0, -2.5], dtype=np.float64)
        target_position_high = np.array([-4.0, 8.5], dtype=np.float64)
        cabin_angle_low = np.array([-np.pi/2, -np.pi/2], dtype=np.float64)
        cabin_angle_high = np.array([np.pi/2, np.pi/2], dtype=np.float64)
        
        # Extend the observation space
        o_low  = np.concatenate([o_low,  bucket_position_low,  rock_position_low,  target_position_low, cabin_angle_low])
        o_high = np.concatenate([o_high, bucket_position_high, rock_position_high, target_position_high, cabin_angle_high])
        
        num_bucket_position_obs = 2
        num_rock_position_obs = 2
        num_target_position_obs = 2
        num_cabin_angle_obs = 2
        Number_of_additional_observations = num_bucket_position_obs +\
                                            num_rock_position_obs +\
                                            num_target_position_obs +\
                                            num_cabin_angle_obs
        ##############################################################
        
        observation_space = spaces.Box(
            low=o_low,
            high=o_high,
            shape=(o_loader.shape[0]+Number_of_additional_observations,),
            dtype=np.float64)
        
        action_space = spaces.Box(
            low=a_low, 
            high=a_high, 
            shape=(a_low.shape[0],), dtype=np.float64)
        return observation_space, action_space
    
    def _observe(self):
        o_loader = self._excavator.observe()
        ##############################################################
        rock_position = self._rock_position_in_under_carriage_body()
        bucket_position = self._bucket_position_in_under_carriage_body()
        euler_angle_in_under_carriage_body = self._euler_ang_in_under_carriage_body()
        targte_position = self._target_position_in_under_carriage_body()  
        o_loader = np.concatenate([o_loader, np.array([bucket_position.x(),
                                                       bucket_position.z(),
                                                       rock_position.x(),
                                                       rock_position.z(),
                                                       targte_position.x(),
                                                       targte_position.z(),
                                                       euler_angle_in_under_carriage_body.x(),
                                                       euler_angle_in_under_carriage_body.y()])])
        ##############################################################     
        o = o_loader
        truncated = self._truncate()
        terminated = self._terminal()
        ##############################################################
        # Return observation, reward, done, info
        info = {}
        # Compute success
        # is_success = terminated  # Success if rock reaches the goal
        is_success = self._successful_condition_2()
        info = {"is_success": is_success}  # Add 'is_success' for success rate calculation
        ##############################################################
        r = self._reward()
        ##############################################################
        arm_prismatic_angle = self._excavator.arm_prismatics[0].getAngle()
        arm_prismatic_speed = self._excavator.arm_prismatics[0].getCurrentSpeed()
        arm_prismatic_force = self._excavator.arm_prismatics[0].getMotor1D().getCurrentForce()

        stick_prismatic_angle = self._excavator.stick_prismatic.getAngle()
        stick_prismatic_speed = self._excavator.stick_prismatic.getCurrentSpeed()
        stick_prismatic_force = self._excavator.stick_prismatic.getMotor1D().getCurrentForce()

        bucket_prismatic_angle = self._excavator.bucket_prismatic.getAngle()
        bucket_prismatic_speed = self._excavator.bucket_prismatic.getCurrentSpeed()
        bucket_prismatic_force = self._excavator.bucket_prismatic.getMotor1D().getCurrentForce()
        
        # Add rock, target, and bucket position to info
        info["rock_position_x"] = rock_position.x()
        info["rock_position_y"] = rock_position.y()
        info["rock_position_z"] = rock_position.z()
        
        info["bucket_position_x"] = bucket_position.x()
        info["bucket_position_y"] = bucket_position.y()
        info["bucket_position_z"] = bucket_position.z()
        
        info["target_position_x"] = targte_position.x()
        info["target_position_y"] = targte_position.y()
        info["target_position_z"] = targte_position.z()
        
        info["chassie_rotation_x"] = euler_angle_in_under_carriage_body.x()
        info["chassie_rotation_y"] = euler_angle_in_under_carriage_body.y()
        
        # Add prismatic joint info
        info["arm_prismatic_angle"] = arm_prismatic_angle
        info["arm_prismatic_speed"] = arm_prismatic_speed
        info["arm_prismatic_force"] = arm_prismatic_force

        info["stick_prismatic_angle"] = stick_prismatic_angle
        info["stick_prismatic_speed"] = stick_prismatic_speed
        info["stick_prismatic_force"] = stick_prismatic_force

        info["bucket_prismatic_angle"] = bucket_prismatic_angle
        info["bucket_prismatic_speed"] = bucket_prismatic_speed
        info["bucket_prismatic_force"] = bucket_prismatic_force
        
        # Add reward info
        info["reward_rock_target_x_axis"] = self._reward_rock_target_x_axis_2()
        info["reward_rock_target_z_axis"] = self._reward_rock_target_z_axis_2() 
        info["reward_rock_bucket_x_axis"] = self._reward_rock_bucket_x_axis_2() 
        info["reward_euler_ang_chassie_body"] = self._reward_euler_ang_under_carriage_body() 
        info["reward_control_input"] = self._reward_control_input()
        info["reward_smoothing_control_input"] = self._reward_smoothing_control_input() 
        info["reward_terminal_condition"] = self._reward_terminal_condition_2()
        
        # Add condition info
        info["condition_rock_target_x_axis"] = 1 if self._condition_rock_target_x_axis_2() else 0
        info["condition_rock_target_z_axis"] = 1 if self._condition_rock_target_z_axis_2() else 0
        info["condition_rock_bucket_x_axis"] = 1 if self._condition_rock_bucket_x_axis_2() else 0
        info["condition_euler_ang_chassie_body"] = 1 if self._condition_euler_ang_under_carriage_body() else 0 
        info["condition_terminal_condition"] = 1 if self._condition_terminal_condition_2() else 0
        
        # Add action info
        info["action_arm"] = self._current_action[0]*self._excavator._max_arm_speed if self._current_action is not None else 0
        info["action_stick"] = self._current_action[1]*self._excavator._max_stick_speed if self._current_action is not None else 0
        info["action_bucket"] = self._current_action[2]*self._excavator._max_bucket_speed if self._current_action is not None else 0
        ##############################################################
        return o, r, terminated, truncated, info
    
    def _reward(self):
        reward_rock_target_x_axis = self._reward_rock_target_x_axis_2()
        reward_rock_target_z_axis = self._reward_rock_target_z_axis_2()
        reward_euler_ang_under_carriage_body = self._reward_euler_ang_under_carriage_body()
        reward_control_input = self._reward_control_input()
        reward_smoothing_control_input = self._reward_smoothing_control_input()
        reward_terminal_condition = self._reward_terminal_condition_2()
        
        reward = reward_rock_target_x_axis+ \
                reward_rock_target_z_axis+ \
                reward_euler_ang_under_carriage_body+ \
                reward_control_input+ \
                reward_smoothing_control_input+\
                reward_terminal_condition
        
                
        return reward
    
    def _truncate(self):
        rock_position = self._rock_position_in_under_carriage_body()
        
        truncated = False
        if self.spec.max_episode_steps is not None:
            truncated = self.episode_step >= self.spec.max_episode_steps
        
        if abs(rock_position[1]) > 1.0 or rock_position[0] < -13.0:
            truncated = True
            
        return truncated
    
    def _successful_condition_2(self):
        successful_condition = False
        condition_terminal = False
        condition_max_time = False
        
        if self.spec.max_episode_steps is not None:
            condition_max_time = self.episode_step >= self.spec.max_episode_steps
        
        if self._condition_terminal_condition_2():
            condition_terminal = True
            
        if condition_terminal and condition_max_time:
            successful_condition = True
            
        return successful_condition
    
    def _reward_rock_target_x_axis_2(self):
        rock_position_x = np.array([self._rock_position_in_under_carriage_body()[0]])
        target_position_x = np.array([self._target_position_in_under_carriage_body()[0]])
        # r = self._reward_error_position(rock_position_x, target_position_x, scale=1.0)
        r = -1*(1/13)*np.linalg.norm(rock_position_x - target_position_x, ord=2)**2  # Euclidean norm (L2)
        return r
    
    def _condition_rock_target_x_axis_2(self):
        rock_position_x = np.array([self._rock_position_in_under_carriage_body()[0]])
        target_position_x = np.array([self._target_position_in_under_carriage_body()[0]])
        condition_rock_target_x_axis = abs(rock_position_x - target_position_x) < 0.20
        return condition_rock_target_x_axis

    def _reward_rock_target_z_axis_2(self):
        rock_position_z = np.array([self._rock_position_in_under_carriage_body()[2]])
        target_position_z = np.array([self._target_position_in_under_carriage_body()[2]])
        
        # r = 0.5*self._reward_error_position(rock_position_z, target_position_z, scale=1.0)
        r = -0.5*(1/4)*np.linalg.norm(rock_position_z - target_position_z, ord=2)**2
        return r
    
    def _condition_rock_target_z_axis_2(self):
        rock_position_z = np.array([self._rock_position_in_under_carriage_body()[2]])
        target_position_z = np.array([self._target_position_in_under_carriage_body()[2]])
        
        condition_rock_target_z_axis = abs(rock_position_z - target_position_z) < 0.20
        return condition_rock_target_z_axis
    
    def _reward_rock_bucket_x_axis_2(self):
        rock_position_x = np.array([self._rock_position_in_under_carriage_body()[0]])
        bucket_position_x = np.array([self._bucket_position_in_under_carriage_body()[0]])
        # r = 0.1*self._reward_error_position(rock_position_x, bucket_position_x, scale=1.0)
        r = -0.1*np.linalg.norm(rock_position_x - bucket_position_x, ord=2)**2
        return r
    
    def _condition_rock_bucket_x_axis_2(self):
        rock_position_x = np.array([self._rock_position_in_under_carriage_body()[0]])
        bucket_position_x = np.array([self._bucket_position_in_under_carriage_body()[0]])
        condition_rock_bucket_x_axis = abs(rock_position_x - bucket_position_x) < 0.5
        return condition_rock_bucket_x_axis
    
    def _reward_euler_ang_under_carriage_body(self):        
        Euler_Ang_under_carriage_body_in_World = self._excavator.under_carriage_body.getRotation().getAsEulerAngles()
        Euler_Ang_under_carriage_body_in_World_x = Euler_Ang_under_carriage_body_in_World.x()
        Euler_Ang_under_carriage_body_in_World_y = Euler_Ang_under_carriage_body_in_World.y()
        # r = 0
        # if np.abs(Euler_Ang_under_carriage_body_in_World_x) > 0.1 or np.abs(Euler_Ang_under_carriage_body_in_World_y) > 0.1:
        #     r = -0.1 * (Euler_Ang_under_carriage_body_in_World_x**2 + Euler_Ang_under_carriage_body_in_World_y**2)
        # r = -0.1 * (Euler_Ang_under_carriage_body_in_World_x**2 + Euler_Ang_under_carriage_body_in_World_y**2)
        r = -1.0 * (Euler_Ang_under_carriage_body_in_World_x**2 + Euler_Ang_under_carriage_body_in_World_y**2)
        return r
    
    def _condition_euler_ang_under_carriage_body(self):        
        Euler_Ang_under_carriage_body_in_World = self._excavator.under_carriage_body.getRotation().getAsEulerAngles()
        Euler_Ang_under_carriage_body_in_World_x = Euler_Ang_under_carriage_body_in_World.x()
        Euler_Ang_under_carriage_body_in_World_y = Euler_Ang_under_carriage_body_in_World.y()
        condition_euler_ang_under_carriage_body = np.abs(Euler_Ang_under_carriage_body_in_World_x) < 0.1 and np.abs(Euler_Ang_under_carriage_body_in_World_y) < 0.1
        return condition_euler_ang_under_carriage_body
    
    def _reward_terminal_condition_2(self):
        r = 0
        rock_terminal_condition_x = self._condition_rock_target_x_axis_2()
        rock_terminal_condition_z = self._condition_rock_target_z_axis_2()
        Euler_Ang_under_carriage_body_terminal_condition = self._condition_euler_ang_under_carriage_body()
        if rock_terminal_condition_x and rock_terminal_condition_z and Euler_Ang_under_carriage_body_terminal_condition:
            r = 5.0 
        return r
    
    def _condition_terminal_condition_2(self):
        rock_terminal_condition_x = self._condition_rock_target_x_axis_2()
        rock_terminal_condition_z = self._condition_rock_target_z_axis_2()
        Euler_Ang_under_carriage_body_terminal_condition = self._condition_euler_ang_under_carriage_body()
        c = rock_terminal_condition_x and rock_terminal_condition_z and Euler_Ang_under_carriage_body_terminal_condition
        
        return c
    
    def _euler_ang_in_under_carriage_body(self):
        Euler_Ang_under_carriage_body_in_World = self._excavator.under_carriage_body.getRotation().getAsEulerAngles()
        # Euler_Ang_under_carriage_body_in_World_x = Euler_Ang_under_carriage_body_in_World.x()
        # Euler_Ang_under_carriage_body_in_World_y = Euler_Ang_under_carriage_body_in_World.y()
        return Euler_Ang_under_carriage_body_in_World
    
    def _rock_position_in_under_carriage_body(self):
        T_under_carriage_body_in_World_inv = self._excavator.under_carriage_body.getTransform().inverse()
        rock_position_in_world = self._rock.getCmPosition()
        rock_position_in_under_carriage_body = T_under_carriage_body_in_World_inv.transformPoint(rock_position_in_world)
        return rock_position_in_under_carriage_body
    
    def _target_position_in_under_carriage_body(self):
        T_under_carriage_body_in_World_inv = self._excavator.under_carriage_body.getTransform().inverse()        
        target_position_in_world = agx.Vec3(self._target_position[0], self._target_position[1], self._target_position[2])
        target_position_in_under_carriage_body = T_under_carriage_body_in_World_inv.transformPoint(target_position_in_world)
        return target_position_in_under_carriage_body
    
    def _bucket_position_in_under_carriage_body(self):
        T_under_carriage_body_in_World_inv = self._excavator.under_carriage_body.getTransform().inverse()
        bucket_position_in_World = self._excavator.bucket_body.getCmPosition()
        bucket_position_in_under_carriage_body = T_under_carriage_body_in_World_inv.transformPoint(bucket_position_in_World)
        return bucket_position_in_under_carriage_body
    
    def _bucket_transformation_in_under_carriage_body(self):
        T_under_carriage_body_in_World_inv = self._excavator.under_carriage_body.getTransform().inverse()
        T_bucket_in_World = self._excavator.bucket_body.getCmTransform()
        T_bucket_in_under_carriage_body = T_bucket_in_World * T_under_carriage_body_in_World_inv
        return T_bucket_in_under_carriage_body
    
    def _generate_random_rock_density(self):
        # not Random
        # low=2000
        # high=2000
        
        # uniform distribution
        # low=2500
        # high=3200
        # density = np.random.uniform(low, high)

        # normal distribution
        mean = 2000
        std_dev = 85  
        density = np.random.normal(mean, std_dev)
        
        return density
    
    def _generate_random_target_position(self):
        # Circle parameters
        center_x = -6.0 #-4.5
        center_z = 2.0
        radius = 0.3 #1.0
        sigma = radius / 3
        
        # # Sample a random point uniformly within a circle
        # angle = np.random.uniform(0, 2 * np.pi)
        # # uniform distribution over the area
        # r = radius * np.sqrt(np.random.uniform(0, 1))  
        # x = center_x + r * np.cos(angle)
        # z = center_z + r * np.sin(angle)
        # y = 0.0
        
        # Sample from 2D normal distribution
        dx, dz = np.random.normal(0, sigma, size=2)

        # Optional: Clip to stay within the 3-sigma circle
        r = np.sqrt(dx**2 + dz**2)
        if r > radius:
            dx *= radius / r
            dz *= radius / r

        x = center_x + dx
        z = center_z + dz
        y = 0.0

        return np.array([x, y, z], dtype=np.float64)
    
    def _generate_random_rock_position(self):
        x_initial = np.random.uniform(-10.5, -7.0)
        # x_initial = np.random.choice(np.arange(-11.0, -7.0 + 0.1, 0.1))
        p_initial = agx.Vec3(x_initial, 0.29, 0.5)
        return p_initial
    
    def _generate_excavator_configuration(self, rock_position):
        
        rock_position_x = rock_position[0]
        config_list  =  [[+0.126, +0.243, -0.882],
                         [+0.081, +0.110, -0.797],
                         [+0.056, -0.028, -0.745],
                         [+0.034, -0.150, -0.745],
                         [-0.007, -0.327, -0.696],
                         [-0.033, -0.392, -0.696],
                         [-0.095, -0.570, -0.696],
                         [-0.099, -0.706, -0.696],
                         [-0.162, -0.797, -0.784]]
        
        if rock_position_x >= -7.0:
            config = config_list[0]
        elif rock_position_x >= -7.5 and rock_position_x < -7.0:
            config = config_list[1]
        elif rock_position_x >= -8.0 and rock_position_x < -7.5:
            config = config_list[2]
        elif rock_position_x >= -8.5 and rock_position_x < -8.0:    
            config = config_list[3]
        elif rock_position_x >= -9.0 and rock_position_x < -8.5:
            config = config_list[4]
        elif rock_position_x >= -9.5 and rock_position_x < -9.0:
            config = config_list[5]
        elif rock_position_x >= -10.0 and rock_position_x < -9.5:
            config = config_list[6]
        elif rock_position_x >= -10.5 and rock_position_x < -10.0:
            config = config_list[7]
        elif rock_position_x < -10.5:
            config = config_list[8]
        
        return config
        
    def _generate_random_geometry_rock(self):
        
        rock_id = random.choice([2, 3, 6])  # randomly pick 2 or 6
        # rock_id = random.randint(2, 6)  # random integer between 2 and 6 (inclusive)
        
        rock_file = f"models/convex_stones/convex_rock{rock_id}.obj"
        mesh_reader = agxIO.MeshReader()
        mesh_reader.readFile(rock_file)
        scale = agx.Vec3(0.001) # rock 2
        vertices = mesh_reader.getVertices()
        scaled_vertices = agx.Vec3Vector()
        for v in vertices:
            scaled_vertices.append(agx.Vec3.mul(v, scale))
        mesh = agxCollide.Convex(scaled_vertices, mesh_reader.getIndices(), "rock")
        new_shape = mesh.shallowCopy()
        
        return new_shape
        
    def _create_visualized_objects(self, sim, root, num_objects=1000, radius=0.05):
        for _ in range(num_objects):
            # position = self._generate_random_target_position()
            position = self._generate_random_rock_position()
            # Create small sphere geometry
            target = agxCollide.Geometry(agxCollide.Sphere(radius))
            target.setEnableCollisions(False)
            target.setPosition(agx.Vec3(*position))
            node = agxOSG.createVisual(target, root)
            agxOSG.setDiffuseColor(node, agxRender.Color.Red())
            agxOSG.setAlpha(node, 0.4)
            sim.add(target)
     
         
    def heuristic_control_policy(self, t):
        return self._excavator.heuristic_control_policy(t)

    def keyboard_control_policy(self, t):
        # enable keybord controls if not already set
        if self._excavator.keyboard_controls is None:
            print("Enabling keyboard controls")
            self._excavator.keyboard_controls = Excavator365Agent.default_keyboard_settings()
            
    
    # def _get_curriculum_threshold(self, progress: float) -> float:
    #     """
    #     Returns the threshold for success based on progress.
        
    #     :param progress: Float between 0 (start) and 1 (end of training)
    #     :return: threshold in meters
    #     """
    #     start_threshold = 1.0
    #     end_threshold = 0.1
    #     return start_threshold * (1 - progress) + end_threshold * progress
    
    # def _condition_bucket_prismatic_angle(self):
    #     c = False
    #     bucket_prismatic_angle = self._excavator.bucket_prismatic.getAngle()
    #     if bucket_prismatic_angle >= 0.2:
    #         c = True
    #     return c
    
    # def _successful_condition_2_without_time(self):
    #     successful_condition = False
    #     condition_terminal = False
        
    #     if self._condition_terminal_condition_2():
    #         condition_terminal = True
            
    #     if condition_terminal:
    #         successful_condition = True
            
    #     return successful_condition

    # def _terminal_with_condition(self):
    #     terminal = False 
    #     if self._condition_terminal_condition_2():
    #         terminal = True 
    #     return terminal
    
    # def _condition_terminal_condition(self):
    #     rock_terminal_condition_x = self._condition_rock_target_x_axis()
    #     rock_terminal_condition_z = self._condition_rock_target_z_axis()
    #     bucket_terminal_condition_x = self._condition_rock_bucket_x_axis()
    #     Euler_Ang_chassie_body_terminal_condition = self._condition_euler_ang_chassie_body()
    #     c = rock_terminal_condition_x and rock_terminal_condition_z and bucket_terminal_condition_x and Euler_Ang_chassie_body_terminal_condition
    #     return c
    
    # def _reward_terminal_condition(self):
    #     r = 0
    #     rock_terminal_condition_x = self._condition_rock_target_x_axis()
    #     rock_terminal_condition_z = self._condition_rock_target_z_axis()
    #     bucket_terminal_condition_x = self._condition_rock_bucket_x_axis()
    #     Euler_Ang_chassie_body_terminal_condition = self._condition_euler_ang_chassie_body()
    #     if rock_terminal_condition_x and rock_terminal_condition_z and bucket_terminal_condition_x and Euler_Ang_chassie_body_terminal_condition:
    #         r = 5.0 
    #     return r
    
    # def _condition_euler_ang_chassie_body(self):        
    #     Euler_Ang_chassie_body_in_World = self._excavator.chassie_body.getRotation().getAsEulerAngles()
    #     Euler_Ang_chassie_body_in_World_x = Euler_Ang_chassie_body_in_World.x()
    #     Euler_Ang_chassie_body_in_World_y = Euler_Ang_chassie_body_in_World.y()
    #     condition_euler_ang_chassie_body = np.abs(Euler_Ang_chassie_body_in_World_x) < 0.1 and np.abs(Euler_Ang_chassie_body_in_World_y) < 0.1
    #     return condition_euler_ang_chassie_body
    
    # def _condition_rock_bucket_x_axis(self):
    #     rock_position_x = np.array([self._rock.getCmPosition().x()])
    #     bucket_position_x = np.array([self._excavator.bucket_body.getCmPosition().x()])
    #     condition_rock_bucket_x_axis = abs(rock_position_x - bucket_position_x) < 0.5
    #     return condition_rock_bucket_x_axis
    
    # def _reward_euler_ang_chassie_body(self):        
    #     Euler_Ang_chassie_body_in_World = self._excavator.chassie_body.getRotation().getAsEulerAngles()
    #     Euler_Ang_chassie_body_in_World_x = Euler_Ang_chassie_body_in_World.x()
    #     Euler_Ang_chassie_body_in_World_y = Euler_Ang_chassie_body_in_World.y()
    #     r = 0
    #     if np.abs(Euler_Ang_chassie_body_in_World_x) > 0.1 or np.abs(Euler_Ang_chassie_body_in_World_y) > 0.1:
    #         r = -0.1 * (Euler_Ang_chassie_body_in_World_x**2 + Euler_Ang_chassie_body_in_World_y**2)
    #     return r
    
    # def _condition_rock_target_z_axis(self):
    #     rock_position_z = np.array([self._rock.getCmPosition().z()])
    #     condition_rock_target_z_axis = abs(rock_position_z - self._target_position_1d_z) < 0.10
    #     return condition_rock_target_z_axis
    
    # def _reward_rock_bucket_x_axis(self):
    #     rock_position_x = np.array([self._rock.getCmPosition().x()])
    #     bucket_position_x = np.array([self._excavator.bucket_body.getCmPosition().x()])
    #     # r = 0.1*self._reward_error_position(rock_position_x, bucket_position_x, scale=1.0)
    #     r = -0.1*np.linalg.norm(rock_position_x - bucket_position_x, ord=2)**2
    #     return r
    
    # def _condition_rock_target_x_axis(self):
    #     rock_position_x = np.array([self._rock.getCmPosition().x()])
    #     condition_rock_target_x_axis = abs(rock_position_x - self._target_position_1d) < 0.10
    #     return condition_rock_target_x_axis
    
    # def _reward_rock_target_z_axis(self):
    #     rock_position_z = np.array([self._rock.getCmPosition().z()])
    #     # r = 0.5*self._reward_error_position(rock_position_z, self._target_position_1d_z, scale=1.0)
    #     r = -0.5*(1/4)*np.linalg.norm(rock_position_z - self._target_position_1d_z, ord=2)**2
    #     return r
    
    # def _reward_rock_target_x_axis(self):
    #     rock_position_x = np.array([self._rock.getCmPosition().x()])
    #     # r = self._reward_error_position(rock_position_x, self._target_position_1d, scale=1.0)
    #     r = -1*(3/13)*np.linalg.norm(rock_position_x - self._target_position_1d, ord=2)**2  # Euclidean norm (L2)
    #     return r
    
    # def _reward_error_position(self, rock_position, target_position, scale=1.0):
    #     """
    #     Compute a reward based on the exponential decay of the Euclidean distance (L2 norm) 
    #     between the rock's position and the target.

    #     Args:
    #         rock_position (np.array): Current position of the rock (x, z).
    #         target_position (np.array): Desired target position of the rock (x, z).
    #         scale (float): Scaling factor to adjust sensitivity of the decay (higher values decay faster).

    #     Returns:
    #         float: Reward value in range (0,1], where 1 is maximum reward (error = 0).
    #     """
    #     error = np.linalg.norm(rock_position - target_position, ord=2)  # Euclidean norm (L2)
    #     #reward = np.exp(-scale * error)  # Exponential decay
    #     reward = 1.0 / (1.0 + error)  # Exponential decay
    #     return reward
    
    # def _successful_condition(self):
    #     successful_condition = False
    #     condition_terminal = False
    #     condition_max_time = False
        
    #     if self.spec.max_episode_steps is not None:
    #         condition_max_time = self.episode_step >= self.spec.max_episode_steps
        
    #     if self._condition_terminal_condition():
    #         condition_terminal = True
            
    #     if condition_terminal and condition_max_time:
    #         successful_condition = True
            
    #     return successful_condition
    
    # def _truncate_old(self):
    #     truncated = False
    #     if self.spec.max_episode_steps is not None:
    #         truncated = self.episode_step >= self.spec.max_episode_steps
        
    #     rock_position = np.array([self._rock.getCmPosition().x(),
    #                               self._rock.getCmPosition().y(),
    #                               self._rock.getCmPosition().z()])
        
    #     if abs(rock_position[1]) > 1.0:
    #         truncated = True
    #     if rock_position[0] < -11.0:
    #         truncated = True
    #     return truncated
    
    # def _reward_old_old(self):
    #     # return self._reward_terrain_mass_in_bucket()
    #     ##############################################################
    #     rock_position = np.array([self._rock.getCmPosition().x(),
    #                                 self._rock.getCmPosition().y(),
    #                                 self._rock.getCmPosition().z()])
    #     rock_position_2d = np.array([self._rock.getCmPosition().x(),
    #                                 self._rock.getCmPosition().z()])
    #     rock_position_1d = np.array([self._rock.getCmPosition().x()])
    #     rock_position_1d_z = np.array([self._rock.getCmPosition().z()])
    #     # print(f'Rock position: {rock_position_1d_z}')
    #     bucket_position = np.array([self._excavator.bucket_body.getCmPosition().x(),
    #                                 self._excavator.bucket_body.getCmPosition().y(),
    #                                 self._excavator.bucket_body.getCmPosition().z()])
    #     bucket_position_2d = np.array([self._excavator.bucket_body.getCmPosition().x(),
    #                                 self._excavator.bucket_body.getCmPosition().z()])
    #     bucket_position_1d = np.array([self._excavator.bucket_body.getCmPosition().x()])
    #     # print(f'bucket position: {bucket_position_1d}')
    #     # target_position_2d = np.array([-7.5, 2.0])
    #     # target_position_1d = np.array([-7.5])
    #     # offset_z =-0.7
    #     Euler_Ang_chassie_body_in_World = self._excavator.chassie_body.getRotation().getAsEulerAngles()
    #     Euler_Ang_chassie_body_in_World_x = Euler_Ang_chassie_body_in_World.x()
    #     Euler_Ang_chassie_body_in_World_y = Euler_Ang_chassie_body_in_World.y()
        
    #     ##############################################################
    #     # ---------------- Initialize reward ----------------
    #     reward = 0

    #     ##############################################################
    #     # ---------------- Terminal Reward ----------------
    #     # rock_terminal_condition = abs(rock_position_1d - target_position_1d) < 0.10
    #     # rock_terminal_condition = np.linalg.norm(rock_position_2d - self._target_position_2d, ord=2) < np.exp(-10.0*self._success_rate)+0.5
    #     # # print(np.exp(-0.001*self.episode_step)+0.5)
    #     # # print("distance",np.linalg.norm(rock_position_2d - target_position_2d, ord=2))
    #     # # bucket_terminal_condition = abs(bucket_position_1d - rock_position_1d) < 0.10
    #     # # bucket_terminal_condition = np.linalg.norm(bucket_position_2d+[0,offset_z] - rock_position_2d, ord=2) < 0.5
    #     # if rock_terminal_condition:
    #     #     # Update success count if this is the first time in the episode
    #     #     if not hasattr(self, '_success_recorded') or not self._success_recorded:
    #     #         self._success_count += 1
    #     #         print('success count:',self._success_count)
    #     #         self._success_rate = self._success_count / self._episode_count if self._episode_count > 0 else 0.0
    #     #         print('success rate:',self._success_rate)
    #     #         self._success_recorded = True
    #     #     # print(f'Rock position: {rock_position_2d}')
    #     #     # print('Rock has been captured @ step:',self.episode_step)
    #     #     reward += 3000.0
    #     #     return reward     
    #     ##############################################################
    #     rock_terminal_condition = abs(rock_position_1d - self._target_position_1d) < 0.10
    #     rock_terminal_condition_z = abs(rock_position_1d_z - self._target_position_1d_z) < 0.10
    #     bucket_terminal_condition = abs(rock_position_1d - bucket_position_1d) < 0.5
    #     cabin_terminal_condition = np.abs(Euler_Ang_chassie_body_in_World_x) < 0.1 and np.abs(Euler_Ang_chassie_body_in_World_y) < 0.1
    #     if rock_terminal_condition and bucket_terminal_condition and cabin_terminal_condition and rock_terminal_condition_z:
    #         time_bonus = 10.0 * (1 - self.episode_step / 1000.0)  # More reward for faster completion
    #         reward += 5.0 #+ time_bonus
    #         # print('Rock has been captured @ step:',self.episode_step)
    #         # return reward
    #     ##############################################################
    #     # reward += self._reward_error_pos(rock_position_1d, target_position_1d, scale=1.0)
    #     # reward += (1.0/1000.0)*self._reward_error_position(rock_position_1d, self._target_position_1d, scale=1.0)
        
        
    #     reward += self._reward_error_position(rock_position_1d, self._target_position_1d, scale=1.0)
    #     # print("rock reward",self._reward_error_position(rock_position_1d, self._target_position_1d, scale=1.0))
    #     reward += 0.5*self._reward_error_position(rock_position_1d_z, self._target_position_1d_z, scale=1.0)
    #     reward += 0.1*self._reward_error_position(rock_position_1d, bucket_position_1d, scale=1.0)
    #     # print("bucket reward",0.1*self._reward_error_position(rock_position_1d, bucket_position_1d, scale=1.0))
    #     # print(abs(rock_position_1d - bucket_position_1d))
    #     # reward += (1.0/1000.0)*self._reward_error_position(rock_position_1d, bucket_position_1d, scale=1.0)
    #     # print("rock_position_1d: ",rock_position_1d)
    #     # print("bucket_position_1d: ",bucket_position_1d)
    #     # reward += self._reward_error_pos(rock_position_1d, bucket_position_1d, scale=1.0)
    #     # print("rock target",self._reward_error_pos(rock_position_1d, target_position_1d, scale=1.0))
    #     # reward += self._reward_error_position(rock_position_2d, bucket_position_2d+[0,offset_z], scale=1.0)
    #     # print("bucket rock",self._reward_error_pos(rock_position_2d, bucket_position_2d+[0,-0.5], scale=1.0))
    #     # print("rock_position_2d: ",rock_position_2d)
    #     # print("bucket_position_2d: ",bucket_position_2d+[0,offset_z])
    #     # print("erros",np.linalg.norm(bucket_position_2d+[0,offset_z] - rock_position_2d, ord=2))
    #     ##############################################################
    #     # if bucket_position[2] < 1.0:
    #     #     reward += -10.0
    #     ##############################################################
    #     if self._current_action is not None:
    #         control_penalty = (0.1) * (1/1.74) * np.linalg.norm(self._current_action, ord=2)
    #     else:
    #         control_penalty = 0.0
    #     # print('control penalty:',control_penalty)
    #     reward -= control_penalty
        
    #     if self._smoothing_penalty is not None:
    #         # print('smoothing penalty:',(1.0) * (1/3.47) * self._smoothing_penalty)
    #         reward -= (1.0) * (1/3.47) * self._smoothing_penalty
        
    #     if np.abs(Euler_Ang_chassie_body_in_World_x) > 0.1 or np.abs(Euler_Ang_chassie_body_in_World_y) > 0.1:
    #         # print("Euler_Ang_chassie_body_in_World_x: ", Euler_Ang_chassie_body_in_World_x)
    #         # print("Euler_Ang_chassie_body_in_World_y: ", Euler_Ang_chassie_body_in_World_y)
    #         # print('tilt penalty',0.1 * (Euler_Ang_chassie_body_in_World_x**2 + Euler_Ang_chassie_body_in_World_y**2))
    #         reward -= 0.1 * (Euler_Ang_chassie_body_in_World_x**2 + Euler_Ang_chassie_body_in_World_y**2)

    #     ##############################################################
    #     # ---------------- Time Penalty (Encourage Faster Completion) ----------------
    #     time_penalty = -1.0/10.0 #-0.001 * self.episode_step  # Small penalty per step
    #     # reward += time_penalty
    #     # print(f"Time Penalty: {time_penalty:.3f}")
    #     ##############################################################
    #     # print(f'Reward: {reward:.3f}')
        
    #     return reward
    
    # def _reward_old(self):
    #     reward_rock_target_x_axis = self._reward_rock_target_x_axis()
    #     reward_rock_target_z_axis = self._reward_rock_target_z_axis()
    #     reward_rock_bucket_x_axis = self._reward_rock_bucket_x_axis()
    #     reward_euler_ang_chassie_body = self._reward_euler_ang_chassie_body()
    #     reward_control_input = self._reward_control_input()
    #     reward_smoothing_control_input = self._reward_smoothing_control_input()
    #     reward_terminal_condition = self._reward_terminal_condition()
        
    #     reward = reward_rock_target_x_axis+ \
    #             reward_rock_target_z_axis+ \
    #             reward_rock_bucket_x_axis+ \
    #             reward_euler_ang_chassie_body+ \
    #             reward_control_input+ \
    #             reward_smoothing_control_input+ \
    #             reward_terminal_condition
        
    #     #############################################################
    #     # print("rock_in_under_carriage_body: ",self._rock_position_in_under_carriage_body())    
    #     # print("bucket_in_under_carriage_body: ",self._bucket_position_in_under_carriage_body())  
    #     #############################################################
    #     return reward
    
    # def _terminal_old_old(self):
    #     rock_position = np.array([self._rock.getCmPosition().x(),
    #                               self._rock.getCmPosition().y(),
    #                               self._rock.getCmPosition().z()])
    #     rock_position_2d = np.array([self._rock.getCmPosition().x(),
    #                               self._rock.getCmPosition().z()])
    #     rock_position_1d = np.array([self._rock.getCmPosition().x()])
    #     rock_position_1d_z = np.array([self._rock.getCmPosition().z()])
        
    #     bucket_position = np.array([self._excavator.bucket_body.getCmPosition().x(),
    #                                 self._excavator.bucket_body.getCmPosition().y(),
    #                                 self._excavator.bucket_body.getCmPosition().z()])
    #     bucket_position_2d = np.array([self._excavator.bucket_body.getCmPosition().x(),
    #                                 self._excavator.bucket_body.getCmPosition().z()])
    #     bucket_position_1d = np.array([self._excavator.bucket_body.getCmPosition().x()])
        
    #     Euler_Ang_chassie_body_in_World = self._excavator.chassie_body.getRotation().getAsEulerAngles()
    #     Euler_Ang_chassie_body_in_World_x = Euler_Ang_chassie_body_in_World.x()
    #     Euler_Ang_chassie_body_in_World_y = Euler_Ang_chassie_body_in_World.y()
        
    #     # offset_z = -0.7
    #     terminal = False
    #     # if abs(self._rock.getCmPosition().z() - 1.5) < 0.05:
    #     #     terminal = True
        
    #     rock_terminal_condition = abs(rock_position_1d - self._target_position_1d) < 0.10
    #     rock_terminal_condition_z = abs(rock_position_1d_z - self._target_position_1d_z) < 0.10
    #     bucket_terminal_condition = abs(rock_position_1d - bucket_position_1d) < 0.5
    #     cabin_terminal_condition = np.abs(Euler_Ang_chassie_body_in_World_x) < 0.1 and np.abs(Euler_Ang_chassie_body_in_World_y) < 0.1
    #     # rock_terminal_condition = np.linalg.norm(rock_position_2d - self._target_position_2d, ord=2) < np.exp(-10.0*self._success_rate)+0.5
    #     # bucket_terminal_condition = np.linalg.norm(bucket_position_2d+[0,offset_z] - rock_position_2d, ord=2) < 0.5
    #     # bucket_terminal_condition = abs(bucket_position_1d - rock_position_1d) < 0.10
    #     ##############################################################
    #     if rock_terminal_condition and bucket_terminal_condition and cabin_terminal_condition and rock_terminal_condition_z:
    #         # print('Terminal condition has been reached @ step:',self.episode_step)
    #         terminal = False #True
    #     return terminal
    
    # def _observe_old(self):
    #     o_loader = self._excavator.observe()
    #     ##############################################################
    #     P_bucket_CM_in_World = self._excavator.bucket_body.getCmPosition()
    #     P_rock_CM_in_World= self._rock.getCmPosition()
    #     Euler_Ang_chassie_body_in_World = self._excavator.chassie_body.getRotation().getAsEulerAngles()
    #     # o_loader = np.concatenate([o_loader, np.array([P_bucket_CM_in_World.x(),
    #     #                                                P_bucket_CM_in_World.y(),
    #     #                                                P_bucket_CM_in_World.z(),
    #     #                                                P_rock_CM_in_World.x(),
    #     #                                                P_rock_CM_in_World.y(),
    #     #                                                P_rock_CM_in_World.z()])])
    #     # o_loader = np.concatenate([o_loader, np.array([P_bucket_CM_in_World.x(),
    #     #                                                P_bucket_CM_in_World.y(),
    #     #                                                P_bucket_CM_in_World.z(),
    #     #                                                P_rock_CM_in_World.x(),
    #     #                                                P_rock_CM_in_World.y(),
    #     #                                                P_rock_CM_in_World.z(),
    #     #                                                self._target_position[0],
    #     #                                                self._target_position[1],
    #     #                                                self._target_position[2]])])
    #     o_loader = np.concatenate([o_loader, np.array([P_bucket_CM_in_World.x(),
    #                                                    P_bucket_CM_in_World.y(),
    #                                                    P_bucket_CM_in_World.z(),
    #                                                    P_rock_CM_in_World.x(),
    #                                                    P_rock_CM_in_World.y(),
    #                                                    P_rock_CM_in_World.z(),
    #                                                    self._target_position[0],
    #                                                    self._target_position[1],
    #                                                    self._target_position[2],
    #                                                    Euler_Ang_chassie_body_in_World.x(),
    #                                                    Euler_Ang_chassie_body_in_World.y()])])
    #     ##############################################################
    #     o = o_loader
        
    #     ##############################################################
    #     truncated = self._truncate()
    #     # truncated = False
    #     # if self.spec.max_episode_steps is not None:
    #     #     truncated = self.episode_step >= self.spec.max_episode_steps
    #     ##############################################################
    #     terminated = self._terminal()
    #     # Return observation, reward, done, info
    #     info = {}
    #     # Compute success
    #     # is_success = terminated  # Success if rock reaches the goal
    #     is_success = self._successful_condition()
    #     info = {"is_success": is_success}  # Add 'is_success' for success rate calculation
    #     ##############################################################
    #     r = self._reward()
    #     ##############################################################
    #     arm_prismatic_angle = self._excavator.arm_prismatics[0].getAngle()
    #     arm_prismatic_speed = self._excavator.arm_prismatics[0].getCurrentSpeed()
    #     arm_prismatic_force = self._excavator.arm_prismatics[0].getMotor1D().getCurrentForce()

    #     stick_prismatic_angle = self._excavator.stick_prismatic.getAngle()
    #     stick_prismatic_speed = self._excavator.stick_prismatic.getCurrentSpeed()
    #     stick_prismatic_force = self._excavator.stick_prismatic.getMotor1D().getCurrentForce()

    #     bucket_prismatic_angle = self._excavator.bucket_prismatic.getAngle()
    #     bucket_prismatic_speed = self._excavator.bucket_prismatic.getCurrentSpeed()
    #     bucket_prismatic_force = self._excavator.bucket_prismatic.getMotor1D().getCurrentForce()
    
    #     # Add rock, target, and bucket position to info
    #     info["rock_position_x"] = P_rock_CM_in_World.x()
    #     info["rock_position_y"] = P_rock_CM_in_World.y()
    #     info["rock_position_z"] = P_rock_CM_in_World.z()
    #     info["bucket_position_x"] = P_bucket_CM_in_World.x()
    #     info["bucket_position_y"] = P_bucket_CM_in_World.y()
    #     info["bucket_position_z"] = P_bucket_CM_in_World.z()
    #     info["target_position_x"] = self._target_position[0]
    #     info["target_position_y"] = self._target_position[1]
    #     info["target_position_z"] = self._target_position[2]
    #     info["chassie_rotation_x"] = Euler_Ang_chassie_body_in_World.x()
    #     info["chassie_rotation_y"] = Euler_Ang_chassie_body_in_World.y()
        
    #     # Add prismatic joint info
    #     info["arm_prismatic_angle"] = arm_prismatic_angle
    #     info["arm_prismatic_speed"] = arm_prismatic_speed
    #     info["arm_prismatic_force"] = arm_prismatic_force

    #     info["stick_prismatic_angle"] = stick_prismatic_angle
    #     info["stick_prismatic_speed"] = stick_prismatic_speed
    #     info["stick_prismatic_force"] = stick_prismatic_force

    #     info["bucket_prismatic_angle"] = bucket_prismatic_angle
    #     info["bucket_prismatic_speed"] = bucket_prismatic_speed
    #     info["bucket_prismatic_force"] = bucket_prismatic_force
        
    #     # Add reward info
    #     info["reward_rock_target_x_axis"] = self._reward_rock_target_x_axis()
    #     info["reward_rock_target_z_axis"] = self._reward_rock_target_z_axis() 
    #     info["reward_rock_bucket_x_axis"] = self._reward_rock_bucket_x_axis() 
    #     info["reward_euler_ang_chassie_body"] = self._reward_euler_ang_chassie_body() 
    #     info["reward_control_input"] = self._reward_control_input()
    #     info["reward_smoothing_control_input"] = self._reward_smoothing_control_input() 
    #     info["reward_terminal_condition"] = self._reward_terminal_condition()
        
    #     # Add condition info
    #     info["condition_rock_target_x_axis"] = 1 if self._condition_rock_target_x_axis() else 0
    #     info["condition_rock_target_z_axis"] = 1 if self._condition_rock_target_z_axis() else 0
    #     info["condition_rock_bucket_x_axis"] = 1 if self._condition_rock_bucket_x_axis() else 0
    #     info["condition_euler_ang_chassie_body"] = 1 if self._condition_euler_ang_chassie_body() else 0 
    #     info["condition_terminal_condition"] = 1 if self._condition_terminal_condition() else 0
        
    #     # Add action info
    #     info["action_arm"] = self._current_action[0]*self._excavator._max_arm_speed if self._current_action is not None else 0
    #     info["action_stick"] = self._current_action[1]*self._excavator._max_stick_speed if self._current_action is not None else 0
    #     info["action_bucket"] = self._current_action[2]*self._excavator._max_stick_speed if self._current_action is not None else 0
    #     # info["action_bucket"] = self._current_action[2]*self._excavator._max_bucket_speed if self._current_action is not None else 0
    #     ##############################################################
    #     return o, r, terminated, truncated, info
    
    # def _setup_gym_environment_spaces_old(self):
    #     o_loader = self._excavator.observe()
    #     o_low, o_high = self._excavator.observation_range()
    #     a_low, a_high = self._excavator.action_range_bucket_arm_boom()

    #     ##############################################################
    #     # Define the low and high values for the bucket and rock position
    #     bucket_position_low = np.array([-20.0, -5.0, -10.0], dtype=np.float64)
    #     bucket_position_high = np.array([20.0, 5.0, 10.0], dtype=np.float64)
    #     rock_position_low = np.array([-20.0, -5.0, -10.0], dtype=np.float64)
    #     rock_position_high = np.array([20.0, 5.0, 10.0], dtype=np.float64)
    #     target_position_low = np.array([-20.0, -5.0, -10.0], dtype=np.float64)
    #     target_position_high = np.array([20.0, 5.0, 10.0], dtype=np.float64)
    #     cabin_angle_low = np.array([-np.pi, -np.pi], dtype=np.float64)
    #     cabin_angle_high = np.array([np.pi, np.pi], dtype=np.float64)
        
    #     # Extend the observation space
    #     # o_low  = np.concatenate([o_low,  bucket_position_low,  rock_position_low])
    #     # o_high = np.concatenate([o_high, bucket_position_high, rock_position_high])
    #     # o_low  = np.concatenate([o_low,  bucket_position_low,  rock_position_low,  target_position_low])
    #     # o_high = np.concatenate([o_high, bucket_position_high, rock_position_high, target_position_high])
    #     o_low  = np.concatenate([o_low,  bucket_position_low,  rock_position_low,  target_position_low, cabin_angle_low])
    #     o_high = np.concatenate([o_high, bucket_position_high, rock_position_high, target_position_high, cabin_angle_high])
        
    #     num_bucket_position_obs = 3
    #     num_bucket_position_obs = 3
    #     num_target_position_obs = 3
    #     num_cabin_angle_obs = 2
    #     # Number_of_additional_observations = num_bucket_position_obs + num_bucket_position_obs + num_target_position_obs
    #     Number_of_additional_observations = num_bucket_position_obs + num_bucket_position_obs + num_target_position_obs + num_cabin_angle_obs
    #     ##############################################################
        
    #     observation_space = spaces.Box(
    #         low=o_low,
    #         high=o_high,
    #         shape=(o_loader.shape[0]+Number_of_additional_observations,),
    #         dtype=np.float64)
        
    #     action_space = spaces.Box(
    #         low=a_low, 
    #         high=a_high, 
    #         shape=(a_low.shape[0],), dtype=np.float64)
    #     return observation_space, action_space
