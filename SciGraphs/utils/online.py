from __future__ import annotations


def online_ok() -> bool:
    """Whether the user allows this session to reach the network.

    True outside Blender, where there is no preference to read and the caller
    is a script that asked for the work explicitly.
    """
    try:
        import bpy
    except ImportError:
        return True
    return bool(getattr(bpy.app, "online_access", True))


def refuse_offline(operator) -> bool:
    """Report the refusal on `operator` and return True when offline.

    For a call that cannot be prevented by `poll` alone, because whether it
    needs the network depends on what the user picked.
    """
    if online_ok():
        return False
    operator.report(
        {'ERROR'},
        "Blender is set to work offline. Enable Preferences > System > "
        "Network > Allow Online Access to use this")
    return True


class OnlineOperator:
    """Mixin for an operator that cannot work without the network.

    Put it first in the bases so its `poll` runs::

        class SCIGRAPHS_OT_Thing(OnlineOperator, bpy.types.Operator):
            ...

    An operator with its own `poll` should call `online_poll(context)` from it
    rather than inherit, or the two conditions will not both be checked.
    """

    @classmethod
    def poll(cls, context):
        return online_ok()

    @classmethod
    def online_poll(cls, context) -> bool:
        return online_ok()

    @classmethod
    def description(cls, context, properties):
        if not online_ok():
            return ("Needs internet access, which is off. Enable Preferences "
                    "> System > Network > Allow Online Access")
        return cls.bl_description if hasattr(cls, "bl_description") else ""


def user_dir(*parts) -> str:
    """A writable directory for caches, created on demand.

    `bpy.utils.extension_path_user` is where an extension is allowed to write.
    Its own directory is not: a "System" repository can sit on a read-only
    filesystem, and the guidelines say so outright. Outside Blender this falls
    back to the platform cache directory so the core keeps working in a plain
    Python process.
    """
    import os

    try:
        import bpy
        root = bpy.utils.extension_path_user(
            __package__.rpartition(".")[0], create=True)
    except Exception:
        root = os.path.join(
            os.environ.get("XDG_CACHE_HOME",
                           os.path.expanduser("~/.cache")), "scigraphs")

    path = os.path.join(root, *parts) if parts else root
    os.makedirs(os.path.dirname(path) if os.path.splitext(path)[1] else path,
                exist_ok=True)
    return path
