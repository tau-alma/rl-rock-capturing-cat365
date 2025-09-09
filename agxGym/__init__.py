from gymnasium.envs.registration import register

from .envs.models.terrains import GravelPile, FlatTerrain
from .envs.models.wheel_loader_agents import WheelLoaderDL300Agent, WheelLoaderWA475Agent, WheelLoaderL70Agent, WheelLoaderAlgoryxAgent

register(
    id='agx-cartpole-v0',
    entry_point='agxPythonModules.agxGym.envs:CartPoleEnv',
    max_episode_steps=200
)

register(
    id='agx-pushing-robot-v0',
    entry_point='agxPythonModules.agxGym.envs:PushingRobotEnv',
    max_episode_steps=300
)

register(
    id='agx-wa475-terrain-v0',
    entry_point='agxPythonModules.agxGym.envs:WheelLoaderTerrainEnv',
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
        "r_coef_energy": -5e-6,
    }
)

register(
    id='agx-wa475-gravel-pile-and-rocks-v0',
    entry_point='agxPythonModules.agxGym.envs:WheelLoaderTerrainEnv',
    max_episode_steps=1200,
    kwargs={
        "wheel_loader_agent_class": WheelLoaderWA475Agent,
        "terrain_model": GravelPile,
        "terrain_model_kwargs": {
            "terrain_size": 30.0,
            "element_size": 0.22,
            "pile_height": 1.4
        },
        "include_rigid_rocks": True,
        "rock_spawner_kwargs": {
            "max_nb_rocks": 3,
            "position_offset": (-4, 0, 0),
            "reset_timer": 4,
            "emitter_size": (1, 1, 1.5)
        },
        "loader_start_position": (-6.5, 0.0, 0.0),
        "loader_start_direction": (1.0, 0.0, 0.0)
    }
)

register(
    id='agx-wa475-flat-terrain-and-rocks-v0',
    entry_point='agxPythonModules.agxGym.envs:WheelLoaderTerrainEnv',
    max_episode_steps=1200,
    kwargs={
        "wheel_loader_agent_class": WheelLoaderWA475Agent,
        "terrain_model": FlatTerrain,
        "include_rigid_rocks": True,
        "rock_spawner_kwargs": {
            "max_nb_rocks": 12,
            "position_offset": (4, 0, 0),
            "reset_timer": 8,
            "emitter_size": (2, 2, 4)
        },
        "terrain_model_kwargs": {
            "element_size": 0.22
        },
        "loader_start_position": (-6.5, 0.0, 0.0),
        "loader_start_direction": (1.0, 0.0, 0.0)
    }
)

register(
    id='agx-dl300-terrain-v0',
    entry_point='agxPythonModules.agxGym.envs:WheelLoaderTerrainEnv',
    max_episode_steps=1200,
    kwargs={
        "wheel_loader_agent_class": WheelLoaderDL300Agent,
        "terrain_model": GravelPile,
        "include_rigid_rocks": False,
        "terrain_model_kwargs": {
            "terrain_size": 30.0,
            "element_size": 0.22,
            "pile_height": 1.4
        },
        "loader_start_position": (-6.5, 0.0, 0.0),
        "loader_start_direction": (1.0, 0.0, 0.0)
    }
)

register(
    id='agx-l70-terrain-v0',
    entry_point='agxPythonModules.agxGym.envs:WheelLoaderTerrainEnv',
    max_episode_steps=1200,
    kwargs={
        "wheel_loader_agent_class": WheelLoaderL70Agent,
        "terrain_model": GravelPile,
        "include_rigid_rocks": False,
        "terrain_model_kwargs": {
            "terrain_size": 30.0,
            "element_size": 0.22,
            "pile_height": 1.4
        },
        "loader_start_position": (-6.5, 0.0, 0.0),
        "loader_start_direction": (1.0, 0.0, 0.0)
    }
)

register(
    id='agx-algoryxwheelloader-terrain-v0',
    entry_point='agxPythonModules.agxGym.envs:WheelLoaderTerrainEnv',
    max_episode_steps=600,
    kwargs={
        "wheel_loader_agent_class": WheelLoaderAlgoryxAgent,
        "terrain_model": GravelPile,
        "include_rigid_rocks": False,
        "include_dump_terrain": False,
        "terrain_model_kwargs": {
            "terrain_size": 15.0,
            "element_size": 0.15,
            "pile_length": 2.5,
            "pile_top": 1,
            "pile_base": 3,
            "pile_height": 0.5
        },
        "loader_start_position": (-4.5, 0.0, 0.0),
        "loader_start_direction": (1.0, 0.0, 0.0)
    }
)

register(
    id='agx-365-terrain-v0',
    entry_point='agxPythonModules.agxGym.envs:ExcavatorTerrainEnv',
    max_episode_steps=1200,
    kwargs={
        "terrain_model_kwargs": {
            "terrain_size_x": 35.0,
            "terrain_size_y": 35.0,
            "element_size": 0.22,
            "terrain_material": "DIRT_1"
        },
    }
)

register(
    id='agx-365-terrain-rock-v0',
    entry_point='agxGym.envs:ExcavatorTerrainEnv',
    max_episode_steps=500,
    kwargs={
        "terrain_model_kwargs": {
            "terrain_size_x": 35.0,#35.0,
            "terrain_size_y": 35.0,#35.0,
            "element_size": 0.22,
            "terrain_material": "DIRT_1"
        },
    }
)
