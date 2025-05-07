import agx
import agxPowerLine
from typing import List
import numpy as np

def reset_bodies(rbs: List[agx.RigidBody], ts: List[agx.AffineMatrix4x4]):
    for rb, t in zip(rbs, ts):
        rb.setLocalTransform(t)
        rb.setVelocity(agx.Vec3())
        rb.setAngularVelocity(agx.Vec3())
    return rbs, ts


def reset_constraints(cs: List[agx.Constraint]):
    for c in cs:
        # Print constraint type and name (if any)
        # print(f"Name: {c.getName()}")
        c.rebind()
        if c.asPrismatic():
            p: agx.Prismatic = c.asPrismatic()
            p.getLock1D().setPosition(0.0)
            p.getMotor1D().setLockedAtZeroSpeed(False)
        elif c.asHinge():
            h = c.asHinge()
            hinge_angle = agx.RotationalAngle.safeCast(h.getAttachmentPair().getAngle(0))
            hinge_angle.setWindingNumber(0)
            h.getLock1D().setPosition(0.0)
            h.getMotor1D().setLockedAtZeroSpeed(False)
    return cs


def reset_powerline(powerline: agxPowerLine):
    for u in powerline.getUnits():
        for i in range(u.getNumActiveDimensions()):
            d = u.getActiveDimension(i)
            rb = d.getOrReserveBody()
            rb.setAngularVelocity(agx.Vec3())
            rb.setVelocity(agx.Vec3())
    return powerline
