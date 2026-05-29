# Reusable Blender API helpers shared across multiple operator modules.

import bpy


def get_or_create_collection(name, parent=None):
    """Return an existing collection or create a new one.

    Args:
        name: Collection name.
        parent: Parent collection.  When ``None`` the scene root is used.
    """
    if name in bpy.data.collections:
        return bpy.data.collections[name]

    collection = bpy.data.collections.new(name)
    if parent:
        parent.children.link(collection)
    else:
        bpy.context.scene.collection.children.link(collection)

    return collection
