# SciGraphs Addon Preferences
#
# User preferences stored in Blender's addon preferences system.
# Access via Edit > Preferences > Add-ons > SciGraphs

import bpy
from bpy.props import (
    StringProperty,
    BoolProperty,
    EnumProperty,
    IntProperty,
    CollectionProperty,
    PointerProperty,
)

# Get the addon package name dynamically
# This ensures bl_idname matches regardless of how the addon is installed
ADDON_PACKAGE = __package__


# Database connection profile property group
class DatabaseConnectionProfile(bpy.types.PropertyGroup):
    """A single database connection profile."""
    
    name: StringProperty(
        name="Profile Name",
        description="Name for this database connection",
        default="New Connection",
    )
    
    db_type: EnumProperty(
        name="Database Type",
        description="Type of database to connect to",
        items=[
            ('POSTGRESQL', "PostgreSQL", "PostgreSQL database"),
            ('MYSQL', "MySQL/MariaDB", "MySQL or MariaDB database"),
            ('SQLITE', "SQLite", "Local SQLite database file"),
            ('SQLSERVER', "SQL Server", "Microsoft SQL Server"),
        ],
        default='POSTGRESQL',
    )
    
    host: StringProperty(
        name="Host",
        description="Database server hostname or IP address",
        default="localhost",
    )
    
    port: IntProperty(
        name="Port",
        description="Database server port",
        default=5432,
        min=1,
        max=65535,
    )
    
    database: StringProperty(
        name="Database",
        description="Name of the database to connect to",
        default="",
    )
    
    username: StringProperty(
        name="Username",
        description="Database username",
        default="",
    )
    
    password: StringProperty(
        name="Password",
        description="Database password (stored in Blender preferences, not in .blend files)",
        subtype='PASSWORD',
        default="",
    )
    
    # SQLite specific
    sqlite_path: StringProperty(
        name="SQLite File",
        description="Path to SQLite database file",
        subtype='FILE_PATH',
        default="",
    )
    
    # SSL/Security options
    use_ssl: BoolProperty(
        name="Use SSL",
        description="Use SSL/TLS for connection",
        default=False,
    )
    
    ssl_mode: EnumProperty(
        name="SSL Mode",
        description="SSL connection mode (PostgreSQL)",
        items=[
            ('disable', "Disable", "No SSL"),
            ('allow', "Allow", "Try SSL, allow non-SSL"),
            ('prefer', "Prefer", "Try SSL first, fallback to non-SSL"),
            ('require', "Require", "Require SSL connection"),
            ('verify-ca', "Verify CA", "Require SSL with CA verification"),
            ('verify-full', "Verify Full", "Require SSL with full verification"),
        ],
        default='prefer',
    )
    
    # Connection timeout
    timeout: IntProperty(
        name="Timeout",
        description="Connection timeout in seconds",
        default=30,
        min=1,
        max=300,
    )
    
    # Use environment variables for credentials
    use_env_password: BoolProperty(
        name="Password from Environment",
        description="Read password from environment variable instead of storing here",
        default=False,
    )
    
    env_password_var: StringProperty(
        name="Environment Variable",
        description="Name of environment variable containing the password",
        default="SCIGRAPHS_DB_PASSWORD",
    )


def get_preferences():
    """
    Get SciGraphs addon preferences.
    
    Returns:
        AddonPreferences instance or None if not found
    """
    addon = bpy.context.preferences.addons.get(ADDON_PACKAGE)
    if addon:
        return addon.preferences
    return None


class SciGraphsPreferences(bpy.types.AddonPreferences):
    """Addon preferences for SciGraphs."""
    
    bl_idname = ADDON_PACKAGE
    
    # OpenTopography API settings
    opentopography_api_key: StringProperty(
        name="OpenTopography API Key",
        description="API key for downloading DEM data from OpenTopography. "
                    "Get a free key at: https://opentopography.org/",
        subtype='PASSWORD',
        default="",
    )
    
    opentopography_default_dataset: EnumProperty(
        name="Default Dataset",
        description="Default DEM dataset to use for downloads",
        items=[
            ('SRTMGL1', "SRTM GL1 (30m)", "NASA SRTM 1 arc-second, global coverage"),
            ('SRTMGL3', "SRTM GL3 (90m)", "NASA SRTM 3 arc-second, faster download"),
            ('AW3D30', "ALOS World 3D (30m)", "JAXA ALOS, good for Asia"),
            ('NASADEM', "NASADEM (30m)", "Improved SRTM from NASA"),
            ('COP30', "Copernicus GLO-30", "EU Copernicus 30m DEM"),
            ('COP90', "Copernicus GLO-90", "EU Copernicus 90m DEM"),
        ],
        default='SRTMGL1',
    )
    
    # Overture Maps API settings
    overture_api_key: StringProperty(
        name="Overture Maps API Key",
        description="API key for downloading Overture Maps data (buildings, POIs, etc.). "
                    "Get a free key at: https://www.overturemapsapi.com/contact",
        subtype='PASSWORD',
        default="DEMO-API-KEY",
    )
    
    # General settings
    auto_apply_material: BoolProperty(
        name="Auto-Apply Terrain Material",
        description="Automatically apply elevation-based material to imported DEMs",
        default=True,
    )
    
    default_dem_method: EnumProperty(
        name="Default Import Method",
        description="Default method for importing DEM terrain",
        items=[
            ('DISPLACE', "Displace (Fast)", "Use subdivision + displace modifier"),
            ('RAW_MESH', "Raw Mesh (Accurate)", "Create mesh from raster pixels"),
        ],
        default='DISPLACE',
    )
    
    default_subdivision_levels: bpy.props.IntProperty(
        name="Default Subdivision Levels",
        description="Default subdivision levels for Displace method",
        default=6,
        min=1,
        max=10,
    )
    
    # OSMnx cache settings
    osmnx_cache_directory: StringProperty(
        name="Cache Directory",
        description="Directory to store cached OSMnx graphs for automatic reloading",
        subtype='DIR_PATH',
        default="",
    )

    # Advanced OSMnx settings (applied via SCIGRAPHS_OT_ApplyOSMnxSettings)
    osmnx_log_console: BoolProperty(
        name="Log to Console",
        description="Pipe OSMnx logs to the system console (ox.settings.log_console)",
        default=True,
    )

    osmnx_use_cache: BoolProperty(
        name="Use OSMnx HTTP Cache",
        description="Cache Overpass responses on disk (ox.settings.use_cache)",
        default=True,
    )

    osmnx_all_oneway: BoolProperty(
        name="Force One-Way Edges (OSM XML export)",
        description="Required when exporting to OSM XML so that each edge is a separate OSM way",
        default=False,
    )

    osmnx_elevation_url_template: StringProperty(
        name="Elevation URL Template",
        description="Custom URL template for Google-style add_node_elevations_google. "
                    "Leave blank to use Open-Elevation. Placeholder: {locations}",
        default="",
    )

    osmnx_useful_tags_way: StringProperty(
        name="Useful Way Tags",
        description="Comma-separated list of OSM way tags to retain (ox.settings.useful_tags_way)",
        default="bridge,tunnel,oneway,lanes,ref,name,highway,maxspeed,service,access,area,landuse,width,est_width,junction",
    )
    
    # Globe Texture API settings
    globe_texture_provider: EnumProperty(
        name="Texture Provider",
        description="Service to use for downloading globe textures",
        items=[
            ('NASA', "NASA API", "NASA EPIC/Blue Marble imagery (free, requires API key)"),
            ('MAPBOX', "Mapbox", "Mapbox satellite tiles (requires API key)"),
            ('MAPTILER', "MapTiler", "MapTiler satellite tiles (free tier available)"),
            ('STADIA', "Stadia Maps", "Stadia Maps tiles (free tier available)"),
            ('PROCEDURAL', "Procedural Only", "Generate textures procedurally (no API needed)"),
        ],
        default='NASA',
    )
    
    nasa_api_key: StringProperty(
        name="NASA API Key",
        description="API key for NASA Earth imagery. Get free key at: https://api.nasa.gov/",
        subtype='PASSWORD',
        default="DEMO_KEY",
    )
    
    mapbox_api_key: StringProperty(
        name="Mapbox Access Token",
        description="Mapbox access token for satellite imagery. Get at: https://mapbox.com/",
        subtype='PASSWORD',
        default="",
    )
    
    maptiler_api_key: StringProperty(
        name="MapTiler API Key",
        description="MapTiler API key for satellite tiles. Free tier at: https://maptiler.com/",
        subtype='PASSWORD',
        default="",
    )
    
    stadia_api_key: StringProperty(
        name="Stadia Maps API Key",
        description="Stadia Maps API key. Free tier at: https://stadiamaps.com/",
        subtype='PASSWORD',
        default="",
    )
    
    # Database connection profiles
    db_profiles: CollectionProperty(
        type=DatabaseConnectionProfile,
        name="Database Connections",
        description="Saved database connection profiles",
    )
    
    active_db_profile_index: IntProperty(
        name="Active Profile",
        description="Index of the active database profile",
        default=0,
        min=0,
    )
    
    def draw(self, context):
        layout = self.layout
        
        # OpenTopography section
        box = layout.box()
        box.label(text="OpenTopography API", icon='URL')
        
        row = box.row()
        row.prop(self, "opentopography_api_key")
        
        # Validation button and status
        row = box.row(align=True)
        row.operator("scigraphs.validate_api_key", text="Validate Key", icon='CHECKMARK')
        row.operator("wm.url_open", text="Get API Key", icon='WORLD').url = "https://opentopography.org/myopentopo"
        
        box.prop(self, "opentopography_default_dataset")
        
        # Help text
        help_box = box.box()
        help_box.scale_y = 0.8
        col = help_box.column(align=True)
        col.label(text="To get an API key:")
        col.label(text="1. Create account at opentopography.org")
        col.label(text="2. Go to MyOpenTopo > API Keys")
        col.label(text="3. Generate a new key and paste here")
        
        layout.separator()
        
        # Overture Maps section
        box = layout.box()
        box.label(text="Overture Maps API", icon='WORLD')
        
        row = box.row()
        row.prop(self, "overture_api_key")
        
        row = box.row(align=True)
        row.operator("wm.url_open", text="Get API Key", icon='WORLD').url = "https://www.overturemapsapi.com/contact"
        
        help_box = box.box()
        help_box.scale_y = 0.8
        col = help_box.column(align=True)
        col.label(text="Overture Maps provides urban data:")
        col.label(text="- Buildings, Places (POIs), Addresses")
        col.label(text="- Transportation infrastructure")
        col.label(text="Default 'DEMO-API-KEY' has rate limits")
        col.label(text="Contact Overture Maps for production API key")
        
        layout.separator()
        
        # Import defaults
        box = layout.box()
        box.label(text="DEM Import Defaults", icon='IMPORT')
        
        box.prop(self, "default_dem_method")
        box.prop(self, "default_subdivision_levels")
        box.prop(self, "auto_apply_material")
        
        layout.separator()
        
        # OSMnx cache settings
        box = layout.box()
        box.label(text="OSMnx Graph Cache", icon='FILE_CACHE')
        
        box.prop(self, "osmnx_cache_directory")
        
        help_box = box.box()
        help_box.scale_y = 0.8
        col = help_box.column(align=True)
        col.label(text="Cache directory stores downloaded OSMnx graphs")
        col.label(text="Graphs are automatically reloaded when Blender restarts")
        col.label(text="Leave empty to use default location in user scripts folder")

        layout.separator()

        # OSMnx advanced settings
        box = layout.box()
        box.label(text="OSMnx Advanced Settings", icon='PREFERENCES')
        col = box.column(align=True)
        col.prop(self, "osmnx_log_console")
        col.prop(self, "osmnx_use_cache")
        col.prop(self, "osmnx_all_oneway")
        col.separator()
        col.prop(self, "osmnx_elevation_url_template")
        col.prop(self, "osmnx_useful_tags_way")
        box.operator("scigraphs.apply_osmnx_settings", text="Apply Now", icon='CHECKMARK')
        
        layout.separator()
        
        # Globe Texture API settings
        box = layout.box()
        box.label(text="Globe Texture APIs", icon='WORLD')
        
        box.prop(self, "globe_texture_provider")
        
        col = box.column(align=True)
        
        if self.globe_texture_provider == 'NASA':
            col.prop(self, "nasa_api_key")
            row = col.row(align=True)
            row.operator("wm.url_open", text="Get Free NASA API Key", icon='URL').url = "https://api.nasa.gov/"
            
            help_box = box.box()
            help_box.scale_y = 0.7
            help_col = help_box.column(align=True)
            help_col.label(text="NASA provides free API keys with 1000 requests/hour")
            help_col.label(text="Default DEMO_KEY has lower limits but works for testing")
            
        elif self.globe_texture_provider == 'MAPBOX':
            col.prop(self, "mapbox_api_key")
            row = col.row(align=True)
            row.operator("wm.url_open", text="Get Mapbox Token", icon='URL').url = "https://account.mapbox.com/access-tokens/"
            
            help_box = box.box()
            help_box.scale_y = 0.7
            help_col = help_box.column(align=True)
            help_col.label(text="Mapbox offers high-quality satellite imagery")
            help_col.label(text="Free tier: 200,000 tile requests/month")
            
        elif self.globe_texture_provider == 'MAPTILER':
            col.prop(self, "maptiler_api_key")
            row = col.row(align=True)
            row.operator("wm.url_open", text="Get MapTiler Key", icon='URL').url = "https://cloud.maptiler.com/account/keys/"
            
            help_box = box.box()
            help_box.scale_y = 0.7
            help_col = help_box.column(align=True)
            help_col.label(text="MapTiler offers satellite and terrain tiles")
            help_col.label(text="Free tier available with registration")
            
        elif self.globe_texture_provider == 'STADIA':
            col.prop(self, "stadia_api_key")
            row = col.row(align=True)
            row.operator("wm.url_open", text="Get Stadia Key", icon='URL').url = "https://client.stadiamaps.com/signup/"
            
            help_box = box.box()
            help_box.scale_y = 0.7
            help_col = help_box.column(align=True)
            help_col.label(text="Stadia Maps offers various map styles")
            help_col.label(text="Free tier: 200,000 tile requests/month")
            
        elif self.globe_texture_provider == 'PROCEDURAL':
            help_box = box.box()
            help_box.scale_y = 0.7
            help_col = help_box.column(align=True)
            help_col.label(text="Procedural textures generated locally")
            help_col.label(text="No API key required, works offline")
        
        layout.separator()
        
        # Database Connections section
        box = layout.box()
        box.label(text="Database Connections", icon='ASSET_MANAGER')
        
        row = box.row()
        row.template_list(
            "SCIGRAPHS_UL_db_profiles", "",
            self, "db_profiles",
            self, "active_db_profile_index",
            rows=3,
        )
        
        col = row.column(align=True)
        col.operator("scigraphs.add_db_profile", icon='ADD', text="")
        col.operator("scigraphs.remove_db_profile", icon='REMOVE', text="")
        
        # Show active profile settings
        if self.db_profiles and len(self.db_profiles) > self.active_db_profile_index:
            profile = self.db_profiles[self.active_db_profile_index]
            
            settings_box = box.box()
            settings_box.label(text="Connection Settings", icon='PREFERENCES')
            
            col = settings_box.column(align=True)
            col.prop(profile, "name")
            col.prop(profile, "db_type")
            
            col.separator()
            
            if profile.db_type == 'SQLITE':
                col.prop(profile, "sqlite_path")
            else:
                col.prop(profile, "host")
                col.prop(profile, "port")
                col.prop(profile, "database")
                col.separator()
                col.prop(profile, "username")
                
                if profile.use_env_password:
                    col.prop(profile, "env_password_var")
                else:
                    col.prop(profile, "password")
                col.prop(profile, "use_env_password")
                
                col.separator()
                col.prop(profile, "use_ssl")
                if profile.use_ssl and profile.db_type == 'POSTGRESQL':
                    col.prop(profile, "ssl_mode")
            
            col.separator()
            col.prop(profile, "timeout")
            
            row = settings_box.row()
            row.operator("scigraphs.test_db_connection", text="Test Connection", icon='PLUGIN')
        
        # Help text for database connections
        help_box = box.box()
        help_box.scale_y = 0.7
        help_col = help_box.column(align=True)
        help_col.label(text="Create database connection profiles here")
        help_col.label(text="Passwords are stored securely in Blender preferences")
        help_col.label(text="Use 'Password from Environment' for extra security")


class SCIGRAPHS_UL_db_profiles(bpy.types.UIList):
    """UI list for database connection profiles."""
    
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            # Icon based on database type
            db_icons = {
                'POSTGRESQL': 'SYSTEM',
                'MYSQL': 'LINKED',
                'SQLITE': 'FILE_BLANK',
                'SQLSERVER': 'DRIVER',
            }
            icon = db_icons.get(item.db_type, 'ASSET_MANAGER')
            layout.prop(item, "name", text="", emboss=False, icon=icon)
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon='ASSET_MANAGER')


class SCIGRAPHS_OT_AddDBProfile(bpy.types.Operator):
    """Add a new database connection profile."""
    bl_idname = "scigraphs.add_db_profile"
    bl_label = "Add Database Profile"
    bl_description = "Add a new database connection profile"
    
    def execute(self, context):
        prefs = get_preferences()
        if prefs:
            profile = prefs.db_profiles.add()
            profile.name = f"Connection {len(prefs.db_profiles)}"
            prefs.active_db_profile_index = len(prefs.db_profiles) - 1
            self.report({'INFO'}, f"Added new database profile: {profile.name}")
        return {'FINISHED'}


class SCIGRAPHS_OT_RemoveDBProfile(bpy.types.Operator):
    """Remove the selected database connection profile."""
    bl_idname = "scigraphs.remove_db_profile"
    bl_label = "Remove Database Profile"
    bl_description = "Remove the selected database connection profile"
    
    def execute(self, context):
        prefs = get_preferences()
        if prefs and prefs.db_profiles:
            idx = prefs.active_db_profile_index
            if 0 <= idx < len(prefs.db_profiles):
                name = prefs.db_profiles[idx].name
                prefs.db_profiles.remove(idx)
                prefs.active_db_profile_index = max(0, idx - 1)
                self.report({'INFO'}, f"Removed database profile: {name}")
        return {'FINISHED'}


class SCIGRAPHS_OT_ValidateAPIKey(bpy.types.Operator):
    """Validate OpenTopography API key."""
    bl_idname = "scigraphs.validate_api_key"
    bl_label = "Validate API Key"
    bl_description = "Test if the API key is valid"
    
    def execute(self, context):
        prefs = get_preferences()
        if not prefs:
            self.report({'ERROR'}, "Could not access addon preferences")
            return {'CANCELLED'}
        
        api_key = prefs.opentopography_api_key
        
        if not api_key:
            self.report({'WARNING'}, "No API key entered")
            return {'CANCELLED'}
        
        from .core.geo.dem_download import validate_api_key
        
        self.report({'INFO'}, "Validating API key...")
        
        result = validate_api_key(api_key)
        
        if result is True:
            self.report({'INFO'}, "API key is valid")
        elif result is False:
            self.report({'ERROR'}, "API key is invalid")
        else:
            self.report({'WARNING'}, "Could not validate key (network error?)")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_ApplyOSMnxSettings(bpy.types.Operator):
    """Apply the OSMnx advanced settings declared in preferences to ox.settings.*."""
    bl_idname = "scigraphs.apply_osmnx_settings"
    bl_label = "Apply OSMnx Settings"
    bl_description = "Apply log_console / use_cache / all_oneway / useful_tags_way / elevation_url_template to osmnx.settings"

    def execute(self, context):
        prefs = get_preferences()
        if not prefs:
            self.report({'ERROR'}, "Could not access preferences")
            return {'CANCELLED'}

        try:
            import osmnx as ox
        except ImportError:
            self.report({'ERROR'}, "OSMnx not installed")
            return {'CANCELLED'}

        try:
            ox.settings.log_console = bool(prefs.osmnx_log_console)
            ox.settings.use_cache = bool(prefs.osmnx_use_cache)
            ox.settings.all_oneway = bool(prefs.osmnx_all_oneway)

            tags = [t.strip() for t in prefs.osmnx_useful_tags_way.split(",") if t.strip()]
            if tags:
                ox.settings.useful_tags_way = tags

            tmpl = prefs.osmnx_elevation_url_template.strip()
            if tmpl:
                ox.settings.elevation_url_template = tmpl

            if prefs.osmnx_cache_directory:
                ox.settings.cache_folder = bpy.path.abspath(prefs.osmnx_cache_directory)

            self.report({'INFO'}, "OSMnx settings applied")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to apply settings: {e}")
            return {'CANCELLED'}


class SCIGRAPHS_OT_TestDBConnection(bpy.types.Operator):
    """Test the selected database connection."""
    bl_idname = "scigraphs.test_db_connection"
    bl_label = "Test Database Connection"
    bl_description = "Test if the database connection works"
    
    def execute(self, context):
        prefs = get_preferences()
        if not prefs or not prefs.db_profiles:
            self.report({'WARNING'}, "No database profiles configured")
            return {'CANCELLED'}
        
        idx = prefs.active_db_profile_index
        if idx < 0 or idx >= len(prefs.db_profiles):
            self.report({'WARNING'}, "No profile selected")
            return {'CANCELLED'}
        
        profile = prefs.db_profiles[idx]
        
        # Import the db_connector module to test connection
        try:
            from .core import db_connector
            success, message = db_connector.test_connection(profile)
            
            if success:
                self.report({'INFO'}, f"Connection successful: {message}")
            else:
                self.report({'ERROR'}, f"Connection failed: {message}")
        except ImportError:
            self.report({'WARNING'}, "Database connector module not available")
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Connection test error: {str(e)}")
            return {'CANCELLED'}
        
        return {'FINISHED'}


classes = [
    DatabaseConnectionProfile,
    SCIGRAPHS_UL_db_profiles,
    SCIGRAPHS_OT_AddDBProfile,
    SCIGRAPHS_OT_RemoveDBProfile,
    SCIGRAPHS_OT_ApplyOSMnxSettings,
    SCIGRAPHS_OT_TestDBConnection,
    SciGraphsPreferences,
    SCIGRAPHS_OT_ValidateAPIKey,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

