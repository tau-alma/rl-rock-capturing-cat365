import math
from typing import List

import agx
import agxSDK
import agxCollide

from .terrains import FlatTerrain

from agxPythonModules.tools.read_file import SimulationFile
from agxPythonModules.tools.simulation_content import DisabledCollisionsStateHandler
from agxPythonModules.utils.environment import simulation, root


class BedTruck(agxSDK.Assembly):
    def __init__(self):
        super().__init__()
        temp_simulation = agxSDK.Simulation()
        data = SimulationFile(simulation=temp_simulation, root=root())

        filename = "BedTruck.agx"

        if not data.load(filename=filename, parent=self):
            raise FileNotFoundError(f'Unable to load: "{filename}"')

        states = {}
        for rb in self.getRigidBodies():
            states[rb.getUuid()] = rb.getEnable()

        disabled_collisions = DisabledCollisionsStateHandler(temp_simulation)
        for disabled in disabled_collisions:
            disabled1 = disabled[0]
            disabled2 = disabled[1] if len(disabled) > 1 else disabled[0]
            if isinstance(disabled1, agxCollide.Geometry):
                continue
            simulation().getSpace().setEnablePair(disabled1, disabled2, False)

        del disabled_collisions
        del data
        del temp_simulation

        for rb in self.getRigidBodies():
            if not rb.getUuid() in states:
                continue
            rb.setEnable(states[rb.getUuid()])
            rb.setMotionControl(agx.RigidBody.STATIC)

        self._colliding_bodies = [
            self.getRigidBody("Bed"),
            self.getRigidBody("TruckBody")
        ]

        self._bed_sensor_geom = agxCollide.Geometry(agxCollide.Box(agx.Vec3(2.4, 1.2, 0.7)))
        self._bed_sensor_geom.setSensor(True)
        self._bed_sensor_geom.setLocalRotation(agx.Quat(math.radians(-4), agx.Vec3().Y_AXIS()))
        self._bed_sensor_geom.setLocalPosition(agx.Vec3(0.2, 0, 0.2))
        self.getRigidBody("Bed").add(self._bed_sensor_geom)

        dump_terrain = FlatTerrain(5.1, 2.6, element_size=0.22)

        bed_body: agx.RigidBody = self.getRigidBody("Bed")
        bed_body.add(dump_terrain.getGeometry())
        dump_terrain.getGeometry().setLocalRotation(agx.Quat(math.radians(-4), agx.Vec3().Y_AXIS()))
        dump_terrain.getGeometry().setLocalPosition(agx.Vec3(0.2, 0, -0.35))
        self._dump_terrain = dump_terrain

    def reset(self):
        self._dump_terrain.reset()

    @property
    def colliding_bodies(self) -> List[agx.RigidBody]:
        return self._colliding_bodies

    @property
    def bed_sensor_geom(self) -> agxCollide.Geometry:
        return self._bed_sensor_geom

    @property
    def dump_terrain(self) -> FlatTerrain:
        return self._dump_terrain
