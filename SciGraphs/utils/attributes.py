# Mesh attribute writes shared across the add-on, on top of databpy.

import databpy as db
import numpy as np

from databpy import AttributeDomains, AttributeTypes

__all__ = ["AttributeDomains", "AttributeTypes", "domain_size", "fit",
           "replace_attribute"]


def domain_size(mesh, domain):
    """Number of elements *mesh* has on *domain*, an ``AttributeDomains``."""
    return len({
        AttributeDomains.POINT: mesh.vertices,
        AttributeDomains.EDGE: mesh.edges,
        AttributeDomains.FACE: mesh.polygons,
        AttributeDomains.CORNER: mesh.loops,
    }[domain])


def fit(values, length, default=0.0):
    """*values* as an array of exactly *length* rows: truncated when longer,
    padded with *default* when shorter. OSMnx meshes carry curve vertices after
    the node vertices, so per-node values are usually short of the vertex count."""
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    out = np.empty((length, *arr.shape[1:]), dtype=np.result_type(arr, np.asarray(default)))
    count = min(length, len(arr))
    out[:count] = arr[:count]
    out[count:] = default
    return out


def replace_attribute(obj, name, values, atype, domain=AttributeDomains.POINT):
    """Store *values* as attribute *name*, removing any existing attribute of
    that name first so its type and domain are replaced too. databpy on its own
    writes into an existing attribute with the type and domain it already has."""
    if name in obj.data.attributes:
        db.remove_named_attribute(obj, name)
    return db.store_named_attribute(obj, values, name, atype=atype, domain=domain)
