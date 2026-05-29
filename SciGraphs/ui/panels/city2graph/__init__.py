from . import main_panel
from . import subpanels
from . import metapath_panels
from . import proximity_panel


def register():
    main_panel.register()
    subpanels.register()
    metapath_panels.register()
    proximity_panel.register()


def unregister():
    proximity_panel.unregister()
    metapath_panels.unregister()
    subpanels.unregister()
    main_panel.unregister()
