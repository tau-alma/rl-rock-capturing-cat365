import agx
import agxOSG
import agxSDK
import agxRender
import agxCollide
import agxTerrain

import numpy as np

import os
import sys
import math
import time
import logging
from typing import List, Tuple
from scipy import ndimage

from agxPythonModules.models.wheel_loaders.wheel_loader import Tire
from agxPythonModules.utils.numpy_utils import wrap_vector_as_numpy_array
from agxPythonModules.utils.environment import simulation

log = logging.getLogger(__name__)


def heightFunction2(x, y, h, size, deltaR=0.2, R=1.15, n=1):
    height = 0

    x /= (size / 2)
    y /= (size / 2)
    x += 0.55

    r = (x**2 + y**2)**0.2
    theta = np.arctan2(x, y)
    theta_threshold = math.pi / 12.0
    theta_min = 50.0 * math.pi / 180.0
    theta_max = math.pi - theta_min

    if (r > R - deltaR and theta > theta_min and theta < theta_max):
        if (r < R):
            height = h * np.power(np.sin((r + deltaR - R) / (2 * deltaR) * math.pi), n)
        else:
            height = h

    if (theta < (theta_min + theta_threshold)):
        theta = theta - theta_min
        height *= np.sin(theta / (2 * theta_threshold) * math.pi)

    if (theta > (theta_max - theta_threshold)):
        theta = theta_max - theta
        height *= np.sin(theta / (2 * theta_threshold) * math.pi)

    return height


def height_field_index_to_position(index, resolution, size):
    return (index / float(resolution) - 0.5) * size


def basic_heights(resolution, size, edge_height=1.0, height=3.8, deltaR=0.2, R=1.15, n=1):
    heights = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            heights[j, i] = heightFunction2(
                height_field_index_to_position(i, resolution, size),
                height_field_index_to_position(j, resolution, size),
                height,
                size,
                deltaR=deltaR,
                R=R,
                n=n)
    edge = np.linspace(0, edge_height, resolution)
    depth = np.linspace(0, edge_height, 8)
    edge_v, depth_v = np.meshgrid(edge, depth)
    heights[0:8, :] = np.flip(depth_v)
    heights[-8:, :] = depth_v
    heights[8:-8, 0:8] = np.flip(depth_v.T)[8:-8, :]
    return heights


def rectangular_pile(size: float = 25.0,
                     element_size: float = 0.11,
                     pile_length: float = 4.0,
                     pile_top: float = 1.2,
                     pile_base: float = 4.2,
                     pile_height: float = 0.8) -> np.ndarray:
    resolution = math.floor(size / element_size) + 1
    half_resolution = math.floor(resolution / 2)

    n_length_half = math.floor(pile_length / (2 * element_size)) + 1
    n_slope_base_half = math.floor(((pile_base - pile_top) / 4) / element_size)
    n_top_half = math.floor(pile_top / (2 * element_size))

    heights = np.zeros((resolution, resolution))

    heights[half_resolution - n_length_half:half_resolution + n_length_half, half_resolution - n_top_half:half_resolution + n_top_half] = pile_height
    x = np.linspace(0, pile_height, 2 * n_slope_base_half)
    y = np.linspace(0, pile_height, n_length_half * 2)
    slope_x, slope_y = np.meshgrid(x, y)

    heights[
        half_resolution - n_length_half:half_resolution + n_length_half,
        half_resolution - n_top_half - 2 * n_slope_base_half:half_resolution - n_top_half] = slope_x

    heights[
        half_resolution - n_length_half:half_resolution + n_length_half,
        half_resolution + n_top_half:half_resolution + n_top_half + 2 * n_slope_base_half] = np.flip(slope_x)

    x = np.linspace(0, pile_height, 2 * n_slope_base_half)
    y = np.linspace(0, pile_height, 2 * n_slope_base_half)
    slope_x, slope_y = np.meshgrid(x, y)

    heights[half_resolution - n_length_half - 2 * n_slope_base_half:half_resolution - n_length_half,
            half_resolution - n_top_half - 2 * n_slope_base_half:half_resolution - n_top_half] = np.minimum(slope_x, slope_y)
    heights[half_resolution - n_length_half - 2 * n_slope_base_half:half_resolution - n_length_half,
            half_resolution + n_top_half:half_resolution + n_top_half + 2 * n_slope_base_half] = np.flip(np.minimum(slope_x, slope_y), axis=1)
    heights[half_resolution + n_length_half:half_resolution + n_length_half + 2 * n_slope_base_half,
            half_resolution - n_top_half - 2 * n_slope_base_half:half_resolution - n_top_half] = np.flip(np.flip(np.minimum(slope_x, slope_y), axis=1))
    heights[half_resolution + n_length_half:half_resolution + n_length_half + 2 * n_slope_base_half,
            half_resolution + n_top_half:half_resolution + n_top_half + 2 * n_slope_base_half] = np.flip(np.minimum(slope_x, slope_y))

    x = np.linspace(0, pile_height, 2 * n_top_half)
    y = np.linspace(0, pile_height, 2 * n_slope_base_half)
    slope_x, slope_y = np.meshgrid(x, y)

    heights[half_resolution - n_length_half - 2 * n_slope_base_half:half_resolution - n_length_half, half_resolution - n_top_half:half_resolution + n_top_half] = slope_y
    heights[half_resolution + n_length_half:half_resolution + n_length_half + 2 * n_slope_base_half, half_resolution - n_top_half:half_resolution + n_top_half] = np.flip(slope_y)
    return heights


def load_numpy_heights(path: str):
    h = np.load(os.path.join(os.path.dirname(__file__), path))
    return h


def load_from_png_heights(path: str, max_height: float):
    from matplotlib import pyplot as plt
    im = plt.imread(os.path.join(os.path.dirname(__file__), path), 'png')
    # if not grayscale image use only red-channel
    if len(im.shape) > 2:
        im = im[:, :, 0]
    # transform image
    im = -1 * (im - 1) * max_height
    return im


def numpy_array_to_agx_vector(arr: np.array):
    v = agx.RealVector()
    for a in np.reshape(arr, (-1)):
        v.append(float(a))
    return v


class TerrainModel(agxTerrain.Terrain):
    def __init__(self,
                 terrain_size_x: float = 35.0,
                 terrain_size_y: float = 35.0,
                 element_size: float = 0.22,
                 terrain_material: str = "DIRT_1"):
        super(TerrainModel, self).__init__(int(terrain_size_x / element_size) + 1, int(terrain_size_y / element_size) + 1, element_size, 2.5)
        self._terrain_size = terrain_size_x
        self._terrain_size = terrain_size_y
        self._element_size = element_size
        self._terrain_material = terrain_material

        # Create the Terrain material type for the terrain
        self.loadLibraryMaterial(self._terrain_material)
        mat = agx.Material('terrain')
        self.setMaterial(mat)
        self.getProperties().setMaximumParticleActivationVolume(4.0) 
        ##################
        # Change the particle radius by a factor.
        # self.getProperties().setSoilParticleSizeScaling(0.5)
        ##################
    
        # get object that contains all the material types in the terrain, particle, heightfield etc.
        terrain_material = self.getTerrainMaterial()
        terrain_material.getParticleProperties().setParticleRestitution(0.0)
        
        # Synchronize now. Otherwise it will not sync until preCollide
        terrain_material.synchronize(self)

        self._init_height_field()
        relaxed_heights = TerrainModel.relax_the_terrain(self)
        self.setHeights(relaxed_heights)

        # Get the buffer to the collision geometries vertices and wrap it as a numpy array.
        # That buffer can we use for observing the terrain height data between steps.
        hf = self.getHeightField()
        mesh_data = hf.getMeshData()
        self.height_field_vertices = mesh_data.getVertices()
        self.np_height_field_vertices = wrap_vector_as_numpy_array(self.height_field_vertices, np.float64)
        self.np_height_field_vertices = np.reshape(self.np_height_field_vertices, (self.getResolutionX(), self.getResolutionY(), 4))

        # An agx::RealVector of the current heights in the terrain
        self.terrain_heights: agx.RealVector = self.getResizedHeightField(1, 1)
        # A numpy array that wraps that RealVector. Can quickly edit that RealVector directly using numpy operations
        self.np_terrain_heights = wrap_vector_as_numpy_array(self.terrain_heights, np.float64)

        self.debug_observe = False

    def create_or_get_renderer(self, root):
        renderer = agxOSG.TerrainVoxelRenderer_find(self, simulation())
        if renderer is not None:
            return renderer
        renderer = agxOSG.TerrainVoxelRenderer(self, root)
        renderer.setRenderHeights(False, agx.RangeReal(-1.25, 1.25))
        renderer.setRenderVoxelSolidMass(False)
        renderer.setRenderVoxelFluidMass(False)
        renderer.setRenderHeightField(True)
        renderer.setRenderVoxelBoundingBox(False)
        renderer.setRenderSoilParticlesMesh(True)
        return renderer

    def _init_height_field(self):
        raise NotImplementedError("This method is implemented in the child class. It must create the initial heightfield of the terrain")

    def reset(self, np_rng: np.random.Generator = None):
        raise NotImplementedError("This method is implemented in the child class. It resets the terrain heightfield following some rule")

    @staticmethod
    def relax_the_terrain(terrain: agxTerrain.Terrain) -> np.ndarray:
        def duplicate_terrain(terrain: agxTerrain.Terrain) -> agxTerrain.Terrain:
            size = terrain.getSize()
            height_field = agxCollide.HeightField(terrain.getResolutionX(), terrain.getResolutionY(), size.x(), size.y())
            height_field.setHeights(terrain.getResizedHeightField(1, 1))
            terrain_duplicate: agxTerrain.Terrain = agxTerrain.Terrain.createFromHeightField(height_field, 2.5)
            return terrain_duplicate

        def relax(terrain: agxTerrain.Terrain) -> agx.RealVector:
            sim = agxSDK.Simulation()
            sim.add(terrain)

            terrain.triggerForceAvalancheAll()
            sim.stepForward()
            modifiedVertices = terrain.getModifiedVertices()
            start = time.time()

            while len(modifiedVertices) > 0:
                sim.stepForward()
                modifiedVertices = terrain.getModifiedVertices()
                if (time.time() - start) > 10.0:
                    sys.exit("ERROR! Terrain relaxation took too long time. " +
                             "You should consider altering the initial terrain with " +
                             "slopes below or close to the angle of repose of your " +
                             "terrain material")
            print(f"Relaxing the terrain took {time.time() - start}")
            return terrain.getResizedHeightField(1, 1)

        relaxed_heights = relax(duplicate_terrain(terrain))

        return relaxed_heights

    def observe(self, frame: agx.Frame = None, o_width: int = -1, o_length: int = -1):
        '''
        Observe the terrain heightmap. The area of the heightmap observed is centered at the
        frame origin in the world coordinate system. And streches out o_length, o_width meters
        in the world direction of the x/y axes of frame.

        If no frame is specified the class self.frame is used.
        If o_width or o_length is set. Then the full terrain size in that dimension is used.
        '''
        if frame is None:
            frame = self.frame

        if o_width == -1:
            o_width = self.getSize().y()

        if o_length == -1:
            o_length = self.getSize().x()

        if isinstance(frame, agx.ObserverFrame):
            frame = frame.getFrame()

        o_length_elements = math.floor(o_length / self._element_size)
        o_width_elements = math.floor(o_width / self._element_size)

        # Get the position and gridpoint of one of the corners
        pos0 = frame.transformPointToWorld(agx.Vec3(-o_length / 2, -o_width / 2, 0))
        ti0 = self.getClosestGridPoint(pos0)

        # Crop the heightfield for performance. The corner will be the new middle point
        s = self.np_height_field_vertices.shape
        crop_length = max(o_length_elements, o_width_elements) + 10
        crop_heights = self.np_height_field_vertices[max(ti0.y() - crop_length, 0):min(ti0.y() + crop_length, s[0]),
                                                     max(ti0.x() - crop_length, 0):min(ti0.x() + crop_length, s[1]), :]
        # Calculate the new middle point
        ti0 = agx.Vec2i(min(ti0.x(), crop_length), min(ti0.y(), crop_length))

        # Pad the heightmap so the corner is in the middle
        # The padded heights are set to constant -1.0. Because of this should the observer frame be
        # placed at or near the ground. If placed above the ground can values outside of
        # of the heightfield be interpreted as just being below frame.
        s = crop_heights.shape
        padX = [max(crop_length - ti0.x(), 0), max(crop_length - (s[1] - ti0.x()), 0)]
        padY = [max(crop_length - ti0.y(), 0), max(crop_length - (s[0] - ti0.y()), 0)]
        padded_heights = np.pad(crop_heights, [padY, padX, [0, 0]], 'constant', constant_values=-1.0)

        # Calculate the angle between the X-Axes
        # TODO: Handle that frame might not be in the plane of the heightfield frame?
        terrain_frame: agx.Frame = self.getGeometry().getFrame()
        terrain_forward = terrain_frame.transformVectorToWorld(agx.Vec3().X_AXIS())
        frame_forward = frame.transformVectorToWorld(agx.Vec3().X_AXIS())
        quat = agx.Quat(terrain_forward, frame_forward)
        angle = quat.getAngle()
        if (agx.Vec3().Z_AXIS() * terrain_forward.cross(frame_forward)) < 0:
            angle = -angle

        # rotate the padded images
        rotated_padded_heights = ndimage.rotate(padded_heights,
                                                math.degrees(angle),
                                                reshape=False,
                                                mode='nearest',
                                                order=0,
                                                prefilter=False,
                                                cval=0.0)

        # slice the rectangle
        heights = rotated_padded_heights[ti0.x() + padX[0]:ti0.x() + padX[0] + o_length_elements,
                                         ti0.y() + padY[0]:ti0.y() + padY[0] + o_width_elements, 2]

        if self.debug_observe:
            agxRender.RenderSingleton.instance().add(pos0, 0.3, agx.Vec4f(0.0, 1.0, 0.0, 1.0))
            for row in rotated_padded_heights[ti0.x() + padX[0]:ti0.x() + padX[0] + o_length_elements,
                                              ti0.y() + padY[0]:ti0.y() + padY[0] + o_width_elements, :]:
                for point in row:
                    wp = self.getGeometry().getFrame().transformPointToWorld(agx.Vec3(point[0], point[1], point[2]))
                    agxRender.RenderSingleton.instance().add(wp, 0.05, agx.Vec4f(1.0, 0.0, 0.0, 1.0))

        # Make sure the heights are relative the height of the frame given
        heights = heights - self.getGeometry().getFrame().transformPointToLocal(frame.getTranslate()).z()

        return heights.flatten()

    def set_new_heights(self, new_heights: np.ndarray):
        heights: agx.RealVector = self.getResizedHeightField(1, 1)
        np_heights = wrap_vector_as_numpy_array(heights, np.float64)
        np_heights[:] = new_heights[:]
        self.setHeights(heights, flipY=False, resetCompaction=True)

    def get_highest_point_under_tires(self, tires: List[Tire]) -> Tuple[agx.Vec3, Tire]:
        tire = tires[0]
        point = agx.Vec3(0, 0, -1e6)
        t = None
        for t in tires:
            p = t.getTireRigidBody().getPosition()
            i = self.getClosestGridPoint(p)
            r = self.getSurfacePositionWorld(i)
            if r.z() > point.z():
                point = r
                tire = t
        return point, tire

    @property
    def frame(self) -> agx.Frame:
        return self.getGeometry().getFrame()

    @property
    def element_size(self) -> float:
        return self._element_size


class GravelPile(TerrainModel):
    def __init__(self,
                 terrain_size: float = 25.0,
                 element_size: float = 0.22,
                 terrain_material: str = "GRAVEL_1",
                 pile_length: float = 5.5,
                 pile_top: float = 2.5,
                 pile_base: float = 7.2,
                 pile_height: float = 2.8):
        self._pile_length = pile_length
        self._pile_top = pile_top
        self._pile_base = pile_base
        self._pile_height = pile_height
        super(GravelPile, self).__init__(terrain_size, terrain_size, element_size, terrain_material=terrain_material)

        self._pile_frame = agx.ObserverFrame()

    def _move_heightfield(self, heights, x_pos, y_pos, theta) -> np.ndarray:
        x_ind = int(x_pos / self._element_size)
        y_ind = int(y_pos / self._element_size)

        m0 = np.asarray([
            [1, 0, self.getResolutionY() / 2],
            [0, 1, self.getResolutionX() / 2],
            [0, 0, 1]
        ])

        m1 = np.asarray([
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta), math.cos(theta), 0],
            [0, 0, 1]
        ])
        m2 = np.asarray([
            [1, 0, -(self.getResolutionY() / 2 + y_ind)],
            [0, 1, -(self.getResolutionX() / 2 + x_ind)],
            [0, 0, 1]
        ])

        moved_heights = ndimage.affine_transform(
            heights.reshape((self.getResolutionX(), self.getResolutionY())),
            m0 @ m1 @ m2
        )
        return moved_heights.flatten()

    def _init_height_field(self):
        new_heights = rectangular_pile(self.getSize().x(), self.getElementSize(), self._pile_length, self._pile_top, self._pile_base, self._pile_height)
        self.set_new_heights(new_heights.flatten())

    def reset(self, np_rng: np.random.Generator):
        # Remove all the soil particles
        ssi = self.getSoilSimulationInterface()
        ssi.getGranularBodySystem().clearAllParticles()

        # move the pile
        x = np_rng.uniform(self._terrain_size / 8 + self._terrain_size / 16, self._terrain_size / 4)
        y = np_rng.uniform(-self._terrain_size / 16, self._terrain_size / 16)
        theta = np_rng.uniform(-math.pi / 8, math.pi / 8)
        moved_heights = self._move_heightfield(
            self.np_terrain_heights,
            x,
            y,
            theta
        )
        self._pile_frame.setLocalPosition(agx.Vec3(x, y, 0.0))
        self._pile_frame.setLocalRotation(agx.Quat(theta, agx.Vec3().Z_AXIS()))

        # reset the new heights
        self.set_new_heights(moved_heights)

    @property
    def frame(self) -> agx.Frame:
        return self._pile_frame.getFrame()


class LargePile(TerrainModel):
    def __init__(self,
                 terrain_size: float = 25.0,
                 element_size: float = 0.11,
                 terrain_material: str = "GRAVEL_1"):
        super(LargePile, self).__init__(terrain_size, terrain_size, element_size, terrain_material=terrain_material)

    def _init_height_field(self):
        new_heights = basic_heights(self.getResolutionX(), self.getSize().x(), edge_height=0.2, height=2.0, deltaR=0.17, R=1.15, n=1)
        self.set_new_heights(new_heights.flatten())

    def reset(self, np_rng: np.random.Generator = None):
        # Remove all the soil particles
        ssi = self.getSoilSimulationInterface()
        ssi.getGranularBodySystem().clearAllParticles()
        self.set_new_heights(self.np_terrain_heights)


class FlatTerrain(TerrainModel):
    def __init__(
        self,
        terrain_size_x: float = 35.0,
        terrain_size_y: float = 35.0,
        element_size: float = 0.22,
        terrain_material: str = "DIRT_1",
        height: float = 0.0,
    ):
        self._height = height
        super(FlatTerrain, self).__init__(terrain_size_x, terrain_size_y, element_size, terrain_material=terrain_material)

    def _init_height_field(self):
        new_heights = np.ones((self.getResolutionX(), self.getResolutionY())) * self._height
        self.set_new_heights(new_heights.flatten())

    def reset(self, np_rng: np.random.Generator = None):
        # Remove all the soil particles
        ssi = self.getSoilSimulationInterface()
        ssi.getGranularBodySystem().clearAllParticles()
        self.set_new_heights(self.np_terrain_heights)


class SlopeTerrain(TerrainModel):
    def __init__(
        self,
        terrain_size_x: float = 25.0,
        terrain_size_y: float = 25.0,
        terrain_material: str = "GRAVEL_1",
        element_size: float = 0.11,
    ):
        super(SlopeTerrain, self).__init__(terrain_size_x, terrain_size_y, element_size, terrain_material=terrain_material)

    def _init_height_field(self):
        h = np.arange(self.getResolutionX() * self.getResolutionY())
        h = np.reshape(h, (self.getResolutionX(), self.getResolutionY())) * 0.001
        self.set_new_heights(h.flatten())

    def reset(self, np_rng: np.random.Generator = None):
        # Remove all the soil particles
        ssi = self.getSoilSimulationInterface()
        ssi.getGranularBodySystem().clearAllParticles()
        self.set_new_heights(self.np_terrain_heights)
