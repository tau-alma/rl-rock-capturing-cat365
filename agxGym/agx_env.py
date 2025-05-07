import sys
import logging
from typing import Optional, Tuple, Union, List, Callable, Sequence, Any, Dict

import numpy as np

import gymnasium as gym
from gymnasium.core import ObsType, ActType, RenderFrame

from agxPythonModules.utils.environment import create_or_set_script_context
from agxPythonModules.sensors.camera_sensors import VirtualCameraSensor

import agx
import agxIO
import agxSDK
import agxOSG

from .utils import DisplayVirtualCameraGUI, ExitException
from stable_baselines3.common.monitor import Monitor

logging.basicConfig()
logger = logging.getLogger(__name__)


class AGXGymEnv(gym.Env):
    _instance_exists: bool = False
    # Create the environment with keyword argument headless=False
    # to create a window that displays graphics.
    _headless: bool = True

    def __new__(cls, *args, **kwargs):
        # Not allowed to create more than one instance in a single process.
        # Throw exception to inform user of this.
        if AGXGymEnv._instance_exists:
            logger.error(
                "There is already an AGXGymEnv running in this processes.\n" +
                "It is not allowed to create another one. If you want to have several \n" +
                "environments running in parallel they must be in separate processes.")
            raise ExitException("Exiting! Due to creating several AGXGymEnv in same process")
        AGXGymEnv._instance_exists = True
        return super(AGXGymEnv, cls).__new__(cls)

    def __init__(self, **kwargs):
        ''' Initializes AGX and creates a simulation. It sets up the gym spaces and builds the scene for the first time. '''
        super().__init__()
        self.init = agx.AutoInit()
        self.closed = False
        self.app = None
        self.sim = agxSDK.Simulation()

        # Create script context
        create_or_set_script_context(self.sim, self.app, None)

        self._build_scene()

        self.simulation_steps_per_action = 1
        self.episode_step = 0

        self.virtual_cameras: List[VirtualCameraSensor] = []
        self.virtual_camera_gui: DisplayVirtualCameraGUI = None
        self.environment_scene_decorator = None

        self.rendered_this_step = False

        self.journal = None

        # We need a timer that we can use for the call to
        # executeOneStepWithGraphics.
        # We will however not use this timer to
        # determine when a rendering frame should be executed as
        # we are not using the graphics throttling
        self.m_timer = agx.HighAccuracyTimer(True)
        self._init_render(**kwargs)

    def __del__(self):
        AGXGymEnv._instance_exists = False
        self.close()

    def close(self):
        if not self.closed:
            if not agx.isShutdown():
                self.sim.cleanup(agxSDK.Simulation.CLEANUP_ALL, False)
            del self.app
            self.app = None
            del self.sim
            self.sim = None
            del self.init
            self.init = None
            self.closed = True

    def start_recording_journal(self, path, journal_config_path=None):
        '''
        Starts recording an agxJournal written to the given path.
        '''
        self.stop_recording_journal()
        if self.journal is None:
            self.journal = agx.Journal(path)
            self.journal.createNewSession()
            self.journal.attach(self.sim, agx.Journal.RECORD)
            if journal_config_path is not None:
                self.journal.loadConfiguration(journal_config_path)

    def stop_recording_journal(self):
        if self.journal is not None:
            self.journal.detach()
            self.journal = None

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        '''
        Gym step. Sets the action and then steps the simulation. After stepping
        the environment is observed and results are returned
        '''
        self._set_action(action)

        if self.app is not None:
            if self.app.done():
                raise ExitException("*** User has closed the application.")
        for _ in range(self.simulation_steps_per_action):
            self.sim.stepForward()

        self.rendered_this_step = False
        obs, reward, terminal, truncated, info = self._observe()
        self.episode_step += 1
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminal, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsType, Dict[str, Any]]:
        '''
        Cleans the simulation and builds the scene again.
        For larger scenes or if the environment resets very often this way of resetting might be very slow.

        Then the environment could implement its own reset method. Typically by manipulating the states of all
        the bodies and in the environment to be in its initial state.
        In that case the child class should make sure to call super(AGXGymEnv, self).reset(seed=seed) which
        sets the seeding correctly
        '''
        super().reset(seed=seed)
        self.sim.cleanup(
            agxSDK.Simulation.SYSTEM |
            agxSDK.Simulation.SPACE |
            agxSDK.Simulation.MATERIALS |
            agxSDK.Simulation.ASSEMBLIES |
            agxSDK.Simulation.PARTICLE_SYSTEMS |
            agxSDK.Simulation.CONTACT_DATA, False)
        self._build_scene()
        self._reset_render()
        obs, _, _, _, d = self._observe()
        self.episode_step = 0
        self.rendered_this_step = False

        return obs, d

    def _reset_render(self):
        ''' Creates the visuals for rendering. '''
        if self.app and self.sim:
            self.app.initSimulation(self.sim, True)

            r = self.app.getSceneRoot()
            for b in self.sim.getRigidBodies():
                agxOSG.createVisual(b, r)

            self._modify_visuals(r)

    def _modify_visuals(self, root):
        pass

    def _setup_virtual_cameras(self, visual_observation_space: bool = False):
        '''
        This method is implemented by the child environment.
        This method creates virtual cameras and has the option to change the observation space of the environment to include the
        image from the camera in the observation of environment
        '''
        return

    def _init_render(self, **kwargs):
        '''
        Creates an ExampleApplication for rendering.
        '''
        if 'render_mode' in kwargs:
            self.render_mode = kwargs['render_mode']
            if self.render_mode not in self.metadata['render_modes'] and self.render_mode is not None:
                logger.warning("Cannot initialize render mode %s for this environment. Please check the render_modes listed for this environment", self.render_mode)
                return
        if 'headless' in kwargs:
            self.headless = kwargs['headless']
        visual_observation_space = False
        if 'visual_observation_space' in kwargs:
            visual_observation_space = kwargs['visual_observation_space']

        if self.render_mode is None:
            return

        if self.app is None:
            self.app = agxOSG.ExampleApplication()
            arguments = [sys.executable]

            # Gym env handles rendering on its own. We do not throttle any graphics
            arguments += ['--targetFPS', '-1']

            if self.headless:
                arguments += ['--osgWindow', '0']

            self.app.init(agxIO.ArgumentParser(arguments))
            # Must turn off autostepping. Otherwise will app.executeOneStepWithGraphics() step simulation aswell
            self.app.setAutoStepping(False)

            # Get the current simulation time step before init application
            # since that resets the timestep
            old_time_step = self.sim.getTimeStep()
            self.app.initSimulation(self.sim, True)
            # Reset the timestep after init the application
            self.sim.setTimeStep(old_time_step)

            create_or_set_script_context(self.sim, self.app, self.app.getSceneRoot())

            self._setup_virtual_cameras(visual_observation_space=visual_observation_space)

            # Create visuals of the scene
            r = self.app.getSceneRoot()
            for b in self.sim.getRigidBodies():
                if b.getEnable():
                    agxOSG.createVisual(b, r)

            self._modify_visuals(r)
        else:
            logger.warning("You are not allowed to initialize graphics twice in the same process!")

    def _build_scene(self):
        ''' This method builds the learning environment. And must be implement by the class that inherits from this class '''
        raise NotImplementedError("Environment must implement this method. It should build or load the simulation model and initialize it.")

    def _observe(self) -> Tuple[ObsType, float, bool, bool, dict]:
        ''' This method returns the observation to the agent. And must be implemented by the class that inherits from this class '''
        raise NotImplementedError("Environment must implement this method. It should return the observation, reward, terminal and info.")

    def _set_action(self, action: ActType):
        ''' This method sets the action values. And must be implemented by the class that inherits from this class '''
        raise NotImplementedError("Environment must implement this method.")

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        '''
            Renders the environment using ExampleApplication OSG rendering in AGX.

            If rendering is initialized in "human" mode:
                .step will render  to the current display or terminal. This mode is typically used together
                with headless=False to create the OSG window to be able to interactively view the environment.

            If rendering is initialized in "rgb_array" mode:
                The image from the first self.virtual_camera in the environment is returned.

            Additional windows showing the virtual_camera views can be created by the user. These
            will also be updated when calling render.
            See agxPythonModules.agxGym.utils.DisplayVirtualCameraGUI() for an example.
        '''

        if self.app:
            if not self.rendered_this_step:
                self.app.executeOneStepWithGraphics(self.m_timer)
                self.rendered_this_step = True

            # Collect the images from the added virtual cameras
            if self.render_mode == "rgb_array":
                # get the image for the first virtual camera
                image = self.virtual_cameras[0].image
            if not self.headless:
                if self.virtual_camera_gui is not None:
                    self.virtual_camera_gui.update()
        else:
            logger.warning("Cannot render without initializing graphics! This should typically be done when initializing the environment.")

        return image if self.render_mode == "rgb_array" else None

    def heuristic_control_policy(self, t: float):
        '''
        This method can be implemented by the user to return action values depending on time.
        '''
        raise NotImplementedError("Environment must implement this method.")

    def keyboard_control_policy(self, t: float):
        '''
        This method can be implemented by the user to set action values using keyboard input
        '''
        raise NotImplementedError("Environment must implement this method.")

    @property
    def headless(self) -> bool:
        return self._headless

    @headless.setter
    def headless(self, value: bool):
        self._headless = value


def make_env(
        environment: Callable[[], AGXGymEnv],
        wrappers: Sequence[gym.Wrapper] = [],
        seed: int = 0,
        render_mode: str = None,
        headless: bool = True,
        visual_observation_space: bool = False) -> Callable[[], AGXGymEnv]:
    """
    Utility function for multiprocessed env.

    :param env: The environment class
    :param seed: (int) the inital seed for RNG
    :param render_mode: render mode
    """
    def _init():
        env = environment(render_mode=render_mode, headless=headless, visual_observation_space=visual_observation_space)
        env.reset(seed=seed, options={})
        for wrapper in wrappers:
            env = wrapper(env)
        return env
    return _init

def make_env_with_Monitor(
        environment: Callable[[], AGXGymEnv],
        wrappers: Sequence[gym.Wrapper] = [],
        seed: int = 0,
        render_mode: str = None,
        headless: bool = True,
        monitor_dir: Optional[str] = None,
        visual_observation_space: bool = False) -> Callable[[], AGXGymEnv]:
    """
    Utility function for multiprocessed env.

    :param env: The environment class
    :param seed: (int) the inital seed for RNG
    :param render_mode: render mode
    """
    def _init():
        env = environment(render_mode=render_mode, headless=headless, visual_observation_space=visual_observation_space)
        env.reset(seed=seed, options={})
        for wrapper in wrappers:
            env = wrapper(env)
        
        # Wrap the environment with Monitor
        env = Monitor(env, monitor_dir)
        
        return env
    return _init

def evaluate_env(env: AGXGymEnv,
                 policy: Callable[[np.ndarray], Tuple[np.ndarray, Optional[np.ndarray]]],
                 render: bool = True,
                 num_episodes: int = -1):
    '''
    Runs the environment with the given policy

    :param env: the environment
    :param policy: the policy that takes the observation latest observation from the environment and returns the next action
    :param num_episodes: The number of episodes to run. -1 means it runs indefinite.
    '''
    n = 0
    try:
        obs = env.reset()
        while n < num_episodes or num_episodes < 0:
            done = False
            episode_reward = 0
            while not done:
                action, _ = policy(obs, deterministic=True)
                obs, r, done, _ = env.step(action)
                episode_reward += r
            n += 1
            print(f"Cumulative reward {episode_reward} for episode {n}")
    except ExitException:
        print("Exited application")

def evaluate_env_extra_variables(env: AGXGymEnv,
                 policy: Callable[[np.ndarray], Tuple[np.ndarray, Optional[np.ndarray]]],
                 render: bool = True,
                 num_episodes: int = -1,
                 save_dir: str = None):
    '''
    Runs the environment with the given policy

    :param env: the environment
    :param policy: the policy that takes the observation latest observation from the environment and returns the next action
    :param num_episodes: The number of episodes to run. -1 means it runs indefinite.
    '''
    all_episode_info = []            # list of info dicts per episode
    all_episode_obs = []         # list of arrays: observations per episode
    
    all_episode_rewards = []     # list of scalars: total reward per episode
    all_episode_rewards_per_timestep = []  # list of arrays: rewards per timestep
    
    n = 0
    try:
        obs = env.reset()
        while n < num_episodes or num_episodes < 0:
            done = False
            episode_reward = 0
            
            episode_obs = []
            episode_info = []  
            episode_rewards = []
            
            while not done:
                episode_obs.append(obs)
                
                action, _ = policy(obs, deterministic=True)
                obs, r, done, info = env.step(action)
                
                # Save the info for the current timestep
                episode_info.append(info)
                
                episode_reward += r
                episode_rewards.append(r)
            
            episode_obs = [np.squeeze(obs) for obs in episode_obs]
            all_episode_obs.append(np.array(episode_obs))
            all_episode_info.append(episode_info)  # Append info for this episode
            
            all_episode_rewards.append(episode_reward)
            all_episode_rewards_per_timestep.append(np.array(episode_rewards)) 
            
            n += 1
            print(f"Cumulative reward {episode_reward} for episode {n}")
        
        # Save data
        np.savez_compressed(
            save_dir,
            episode_rewards=np.array(all_episode_rewards),  
            episode_rewards_per_timestep=np.array(all_episode_rewards_per_timestep, dtype=object),
            episode_observations=np.array(all_episode_obs, dtype=object),
            episode_info=np.array(all_episode_info, dtype=object)
        )   
        
        print(f"\n Saved evaluation data to {save_dir}")
    except ExitException:
        print("Exited application")
