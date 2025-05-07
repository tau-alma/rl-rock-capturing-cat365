import agx
import agxCollide

from agxPythonModules.utils.callbacks import ContactEventCallback as cec, StepEventCallback as sec

from typing import List, Tuple, Union


class ContactSensor():
    '''
    Sets up ContactEventListeners between each of the colliders in colliders0
    against every other collider in colliders1.

    If any of the colliders in colliders0 are in contact with any colliders in colliders1
    then the ContactSensor.contact flag will be set.

    The number of such collisions each timestep is counted in ContactSensor.nb_contacts

    The actual rigid bodies in contact are saved as contact pairs in the ContactSensor.contacts_pairs list.
    '''
    def __init__(self,
                 colliders0: List[Union[agx.RigidBody, agxCollide.Geometry]],
                 colliders1: List[Union[agx.RigidBody, agxCollide.Geometry]]):
        self._colliders0 = []
        self._colliders1 = []
        self._contact = False
        self._n = 0
        self._contact_pairs = []

        self.add_colliders(colliders0, colliders1)

    def add_colliders(self,
                      colliders0: List[Union[agx.RigidBody, agxCollide.Geometry]],
                      colliders1: List[Union[agx.RigidBody, agxCollide.Geometry]]):
        self._colliders0 += colliders0
        self._colliders1 += colliders1
        for c in colliders0:
            cec.postCallback(c, *colliders1, self._on_contact_callback)
        sec.preCollideCallback(self._pre_collide_callback)

    def clear(self):
        cec.remove(self._on_contact_callback)
        sec.remove(self._pre_collide_callback)
        self._colliders0 = []
        self._colliders1 = []
        self._contact_pairs = []

    def _pre_collide_callback(self, t):
        self._contact = False
        self._n = 0
        self._contact_pairs = []

    def _on_contact_callback(self, t, gc):
        self._n += 1
        self._contact = True
        self._contact_pairs.append((gc.rigidBody(0), gc.rigidBody(1)))

    @property
    def contact(self) -> bool:
        return self._contact

    @property
    def nb_contacts(self) -> int:
        return self._n

    @property
    def contact_pairs(self) -> List[Tuple[agx.RigidBody, agx.RigidBody]]:
        return self._contact_pairs

    def in_contact_pairs(self, rb):
        '''
        Returns true if rb is one of the bodies that is currently in the
        contact pairs.
        '''
        for rb0, rb1 in self.contact_pairs:
            if rb == rb0:
                return True
            if rb == rb1:
                return True
        return False
