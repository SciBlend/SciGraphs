# Visualization and appearance operators

import bpy
from ....core import geometry
from ....core.mesh.geometry import (
    DEFAULT_NODE_SIZE,
    DEFAULT_NODE_RESOLUTION,
    DEFAULT_NODE_ATTR_MULT,
    DEFAULT_EDGE_THICKNESS,
    DEFAULT_EDGE_RESOLUTION,
    DEFAULT_EDGE_ATTR_MULT,
    NODE_SHAPE_INDEX_MAP,
    _apply_node_primitive_inputs,
)


class SCIGRAPHS_OT_UpdateAppearance(bpy.types.Operator):
    """Update visual appearance of graph."""
    bl_idname = "scigraphs.update_appearance"
    bl_label = "Update Appearance"
    bl_description = "Update the visual appearance of the graph"
    bl_options = {'UNDO'}

    node_size: bpy.props.FloatProperty(
        name="Node Size",
        description="Radius of node primitives",
        default=DEFAULT_NODE_SIZE,
        min=0.0,
        max=1.0,
        options={'SKIP_SAVE'},
    )

    node_resolution: bpy.props.IntProperty(
        name="Node Resolution",
        description="Resolution of generated node geometry",
        default=DEFAULT_NODE_RESOLUTION,
        min=3,
        max=128,
        options={'SKIP_SAVE'},
    )

    node_shape: bpy.props.EnumProperty(
        name="Node Shape",
        description="Generated mesh shape for the graph nodes",
        options={'SKIP_SAVE'},
        items=[
            ('SPHERE', "Sphere", "UV sphere nodes"),
            ('ICOSPHERE', "Icosphere", "Icosphere nodes (lower-poly, more uniform)"),
            ('CUBE', "Cube", "Cube nodes"),
            ('CONE', "Cone", "Cone nodes"),
            ('CYLINDER', "Cylinder", "Cylinder nodes"),
        ],
        default='SPHERE',
    )

    node_scale_multiplier: bpy.props.FloatProperty(
        name="Node Attr Mult",
        description="Multiplier for attribute-driven node scaling",
        default=DEFAULT_NODE_ATTR_MULT,
        min=0.0,
        max=100.0,
        options={'SKIP_SAVE'},
    )

    edge_thickness: bpy.props.FloatProperty(
        name="Edge Thickness",
        description="Thickness of edge tubes (0 = invisible edges)",
        default=DEFAULT_EDGE_THICKNESS,
        min=0.0,
        max=0.5,
        options={'SKIP_SAVE'},
    )

    edge_resolution: bpy.props.IntProperty(
        name="Edge Resolution",
        description="Resolution of edge tube profiles",
        default=DEFAULT_EDGE_RESOLUTION,
        min=3,
        max=64,
        options={'SKIP_SAVE'},
    )

    edge_thickness_multiplier: bpy.props.FloatProperty(
        name="Edge Attr Mult",
        description="Multiplier for attribute-driven edge thickness",
        default=DEFAULT_EDGE_ATTR_MULT,
        min=0.0,
        max=100.0,
        options={'SKIP_SAVE'},
    )

    show_arrows: bpy.props.BoolProperty(
        name="Show Direction Arrows",
        description="Display arrows for directed graph edges when supported",
        default=False,
        options={'SKIP_SAVE'},
    )

    arrow_size: bpy.props.FloatProperty(
        name="Arrow Size",
        description="Size of direction arrows",
        default=0.15,
        min=0.001,
        max=5.0,
        options={'SKIP_SAVE'},
    )

    arrow_position: bpy.props.FloatProperty(
        name="Arrow Position",
        description="Position of direction arrows along each edge",
        default=0.7,
        min=0.0,
        max=1.0,
        options={'SKIP_SAVE'},
    )

    def invoke(self, context, event):
        props = context.scene.scigraphs
        for name in self._appearance_property_names():
            if hasattr(props, name):
                setattr(self, name, getattr(props, name))
        return context.window_manager.invoke_props_dialog(self, width=460)

    def check(self, context):
        self._apply_appearance(context, report=False)
        return True

    def draw(self, context):
        ui_layout = self.layout
        ui_layout.use_property_split = True
        ui_layout.use_property_decorate = False

        node_box = ui_layout.box()
        node_box.use_property_decorate = False
        node_box.label(text="Nodes", icon='MESH_UVSPHERE')
        col = node_box.column(align=True)
        col.use_property_decorate = False
        col.prop(self, "node_size", slider=True)
        col.prop(self, "node_resolution")
        col.prop(self, "node_shape")
        col.prop(self, "node_scale_multiplier")

        edge_box = ui_layout.box()
        edge_box.use_property_decorate = False
        edge_box.label(text="Edges", icon='CURVE_PATH')
        col = edge_box.column(align=True)
        col.use_property_decorate = False
        col.prop(self, "edge_thickness", slider=True)
        col.prop(self, "edge_resolution")
        col.prop(self, "edge_thickness_multiplier")

        arrow_box = ui_layout.box()
        arrow_box.use_property_decorate = False
        arrow_box.label(text="Direction Arrows", icon='FORWARD')
        col = arrow_box.column(align=True)
        col.use_property_decorate = False
        col.prop(self, "show_arrows")
        if self.show_arrows:
            col.prop(self, "arrow_size")
            col.prop(self, "arrow_position", slider=True)

    @staticmethod
    def _appearance_property_names():
        return (
            "node_size",
            "node_resolution",
            "node_shape",
            "node_scale_multiplier",
            "edge_thickness",
            "edge_resolution",
            "edge_thickness_multiplier",
            "show_arrows",
            "arrow_size",
            "arrow_position",
        )

    def execute(self, context):
        result = self._apply_appearance(context, report=True)
        return result

    def _apply_appearance(self, context, report=False):
        props = context.scene.scigraphs
        obj = context.active_object

        if not obj or "num_nodes" not in obj:
            if report:
                self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}

        mod = obj.modifiers.get("SciGraphs_Viz")
        if not mod or not mod.node_group:
            if report:
                self.report({'ERROR'}, "No SciGraphs modifier found")
            return {'CANCELLED'}

        for name in self._appearance_property_names():
            if hasattr(props, name):
                setattr(props, name, getattr(self, name))

        shape_index = NODE_SHAPE_INDEX_MAP.get(self.node_shape, 0)

        # Persist the choices on the object so rebuilds (centrality, layouts,
        # edge styles, ...) restore the same look instead of going back to
        # whatever was hard-coded in the GN tree builders.
        obj["scigraphs_node_size"] = float(self.node_size)
        obj["scigraphs_node_resolution"] = int(self.node_resolution)
        obj["scigraphs_node_shape_index"] = int(shape_index)
        obj["scigraphs_edge_thickness"] = float(self.edge_thickness)
        obj["scigraphs_edge_resolution"] = int(self.edge_resolution)

        node_group = mod.node_group

        # Older trees were built before the multi-primitive shape switch /
        # smooth-by-angle existed. Rebuild on the fly so users don't need to
        # re-run Setup Viz manually after an addon upgrade.
        tree_name = node_group.name if node_group else ""
        is_simple_tree = not tree_name.startswith("SciGraphs_Interactive")
        needs_upgrade = is_simple_tree and (
            node_group.nodes.get("SciGraphs_NodeShapeSwitch") is None
            or node_group.nodes.get("SciGraphs_NodeSmoothByAngle") is None
        )
        if needs_upgrade:
            # Use the public rebuild path so post-rebuild hooks (e.g. coloring)
            # also re-apply themselves to the new tree.
            geometry._rebuild_visualization_if_present(obj)  # pylint: disable=protected-access
            node_group = mod.node_group

        updated = 0

        # Interactive tree exposes everything as modifier sockets. Setting
        # them is a no-op for the simple tree (no matching interface socket).
        socket_values = {
            "Node Scale": self.node_size,
            "Node Resolution": self.node_resolution,
            "Node Shape": shape_index,
            "Node Attr Mult": self.node_scale_multiplier,
            "Edge Thickness": self.edge_thickness,
            "Edge Resolution": self.edge_resolution,
            "Edge Attr Mult": self.edge_thickness_multiplier,
            "Show Arrows": self.show_arrows,
            "Arrow Size": self.arrow_size,
            "Arrow Position": self.arrow_position,
        }
        for socket_name, value in socket_values.items():
            if self._set_modifier_input(mod, socket_name, value):
                updated += 1

        # Direct node tweaks (covers the simple tree where there are no
        # interface sockets, and complements the interactive tree for size).
        if not node_group.nodes.get("SciGraphs_NodeSphere"):
            geometry.fix_node_names(obj)

        primitive_names = {
            'SciGraphs_NodeSphere',
            'SciGraphs_NodeIcoSphere',
            'SciGraphs_NodeCube',
            'SciGraphs_NodeCone',
            'SciGraphs_NodeCylinder',
        }
        for node_name in primitive_names:
            node = node_group.nodes.get(node_name)
            if node is None:
                continue
            updated += _apply_node_primitive_inputs(
                node, self.node_size, self.node_resolution
            )

        shape_switch_node = node_group.nodes.get("SciGraphs_NodeShapeSwitch")
        if shape_switch_node and 'Index' in shape_switch_node.inputs:
            shape_switch_node.inputs['Index'].default_value = shape_index
            updated += 1

        edge_node = node_group.nodes.get("SciGraphs_EdgeProfile")
        if edge_node and 'Radius' in edge_node.inputs:
            edge_node.inputs['Radius'].default_value = self.edge_thickness
            updated += 1
        if edge_node and 'Resolution' in edge_node.inputs:
            edge_node.inputs['Resolution'].default_value = max(3, int(self.edge_resolution))
            updated += 1

        if updated > 0:
            if obj.data:
                obj.data.update()
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
            if report:
                self.report({'INFO'}, f"Updated {updated} parameters")
        else:
            if report:
                self.report({'WARNING'}, "Could not find nodes to update")

        return {'FINISHED'}

    @staticmethod
    def _set_modifier_input(mod, socket_name, value):
        node_group = mod.node_group
        if not node_group:
            return False

        for item in node_group.interface.items_tree:
            if item.item_type == 'SOCKET' and item.in_out == 'INPUT' and item.name == socket_name:
                try:
                    mod[item.identifier] = value
                    return True
                except (KeyError, TypeError, AttributeError):
                    return False
        return False




class SCIGRAPHS_OT_ApplyRenderingPreset(bpy.types.Operator):
    """Apply rendering preset (materials, lighting)."""
    bl_idname = "scigraphs.apply_rendering_preset"
    bl_label = "Apply Rendering Preset"
    bl_description = "Apply pre-configured rendering settings"
    bl_options = {'REGISTER', 'UNDO'}
    
    preset: bpy.props.EnumProperty(
        name="Preset",
        items=[
            ('BASIC', "Basic", "Simple diffuse material"),
            ('GLASS', "Glass", "Transparent glass-like"),
            ('METALLIC', "Metallic", "Metallic reflective"),
            ('EMISSION', "Emission", "Glowing emissive"),
            ('SCIENTIFIC', "Scientific", "Clean scientific"),
        ],
        default='BASIC',
    )
    
    def invoke(self, context, event):
        props = context.scene.scigraphs
        self.preset = props.rendering_preset
        return context.window_manager.invoke_props_dialog(self)
    
    def execute(self, context):
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        
        preset = self.preset
        
        if preset == 'BASIC':
            self._apply_basic_preset(obj, context)
        elif preset == 'GLASS':
            self._apply_glass_preset(obj, context)
        elif preset == 'METALLIC':
            self._apply_metallic_preset(obj, context)
        elif preset == 'EMISSION':
            self._apply_emission_preset(obj, context)
        elif preset == 'SCIENTIFIC':
            self._apply_scientific_preset(obj, context)
        
        self.report({'INFO'}, f"Applied {preset} rendering preset")
        return {'FINISHED'}
    
    def _apply_basic_preset(self, obj, context):
        """Apply basic Principled BSDF material."""
        if not obj.data.materials:
            mat = bpy.data.materials.new(name="SciGraphs_Basic")
            mat.use_nodes = True
            obj.data.materials.append(mat)
        else:
            mat = obj.data.materials[0]
        
        # Configure Principled BSDF
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Metallic'].default_value = 0.0
            bsdf.inputs['Roughness'].default_value = 0.5
    
    def _apply_glass_preset(self, obj, context):
        """Apply glass/transparent material."""
        if not obj.data.materials:
            mat = bpy.data.materials.new(name="SciGraphs_Glass")
            mat.use_nodes = True
            obj.data.materials.append(mat)
        else:
            mat = obj.data.materials[0]
        
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            # Blender 4.5+ uses 'Transmission Weight' instead of 'Transmission'
            if 'Transmission Weight' in bsdf.inputs:
                bsdf.inputs['Transmission Weight'].default_value = 0.9
            elif 'Transmission' in bsdf.inputs:
                bsdf.inputs['Transmission'].default_value = 0.9
            
            bsdf.inputs['Roughness'].default_value = 0.0
            
            # IOR may also have changed names
            if 'IOR' in bsdf.inputs:
                bsdf.inputs['IOR'].default_value = 1.45
    
    def _apply_metallic_preset(self, obj, context):
        """Apply metallic material."""
        if not obj.data.materials:
            mat = bpy.data.materials.new(name="SciGraphs_Metallic")
            mat.use_nodes = True
            obj.data.materials.append(mat)
        else:
            mat = obj.data.materials[0]
        
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Metallic'].default_value = 1.0
            bsdf.inputs['Roughness'].default_value = 0.2
    
    def _apply_emission_preset(self, obj, context):
        """Apply emission material (glowing)."""
        if not obj.data.materials:
            mat = bpy.data.materials.new(name="SciGraphs_Emission")
            mat.use_nodes = True
            obj.data.materials.append(mat)
        else:
            mat = obj.data.materials[0]
        
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            # Blender 4.5+ uses 'Emission Color' instead of 'Emission'
            if 'Emission Color' in bsdf.inputs:
                bsdf.inputs['Emission Color'].default_value = (1.0, 1.0, 1.0, 1.0)
            elif 'Emission' in bsdf.inputs:
                bsdf.inputs['Emission'].default_value = (1.0, 1.0, 1.0, 1.0)
            
            if 'Emission Strength' in bsdf.inputs:
                bsdf.inputs['Emission Strength'].default_value = 2.0
    
    def _apply_scientific_preset(self, obj, context):
        """Apply clean scientific visualization material."""
        if not obj.data.materials:
            mat = bpy.data.materials.new(name="SciGraphs_Scientific")
            mat.use_nodes = True
            obj.data.materials.append(mat)
        else:
            mat = obj.data.materials[0]
        
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Metallic'].default_value = 0.0
            bsdf.inputs['Roughness'].default_value = 0.3
            # Blender 4.5+ changed 'Specular' to 'Specular IOR Level'
            if 'Specular IOR Level' in bsdf.inputs:
                bsdf.inputs['Specular IOR Level'].default_value = 0.5
            elif 'Specular' in bsdf.inputs:
                bsdf.inputs['Specular'].default_value = 0.5


class SCIGRAPHS_OT_SetupLighting(bpy.types.Operator):
    """Setup lighting for visualization."""
    bl_idname = "scigraphs.setup_lighting"
    bl_label = "Setup Lighting"
    bl_description = "Add lights for better visualization"
    bl_options = {'REGISTER', 'UNDO'}
    
    lighting_type: bpy.props.EnumProperty(
        name="Lighting",
        items=[
            ('THREE_POINT', "3-Point", "Classic 3-point lighting"),
            ('STUDIO', "Studio", "Soft studio lighting"),
            ('OUTDOOR', "Outdoor", "Outdoor sun lighting"),
        ],
        default='THREE_POINT',
    )
    
    def invoke(self, context, event):
        props = context.scene.scigraphs
        self.lighting_type = props.lighting_setup
        return context.window_manager.invoke_props_dialog(self)
    
    def execute(self, context):
        # Remove existing SciGraphs lights
        for obj in bpy.data.objects:
            if obj.name.startswith("SciGraphs_Light"):
                bpy.data.objects.remove(obj, do_unlink=True)
        
        lighting_type = self.lighting_type
        
        if lighting_type == 'THREE_POINT':
            self._create_three_point_lighting(context)
        elif lighting_type == 'STUDIO':
            self._create_studio_lighting(context)
        elif lighting_type == 'OUTDOOR':
            self._create_outdoor_lighting(context)
        
        self.report({'INFO'}, f"Applied {lighting_type} lighting setup")
        return {'FINISHED'}
    
    def _create_three_point_lighting(self, context):
        """Create classic 3-point lighting."""
        # Key light
        bpy.ops.object.light_add(type='SUN', location=(5, -5, 8))
        key = context.active_object
        key.name = "SciGraphs_Light_Key"
        key.data.energy = 2.0
        
        # Fill light
        bpy.ops.object.light_add(type='AREA', location=(-5, -3, 5))
        fill = context.active_object
        fill.name = "SciGraphs_Light_Fill"
        fill.data.energy = 0.5
        
        # Rim light
        bpy.ops.object.light_add(type='SPOT', location=(0, 5, 3))
        rim = context.active_object
        rim.name = "SciGraphs_Light_Rim"
        rim.data.energy = 1.0
    
    def _create_studio_lighting(self, context):
        """Create soft studio lighting."""
        bpy.ops.object.light_add(type='AREA', location=(0, 0, 10))
        light = context.active_object
        light.name = "SciGraphs_Light_Top"
        light.data.energy = 3.0
        light.data.size = 10.0
    
    def _create_outdoor_lighting(self, context):
        """Create outdoor-style lighting."""
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
        sun = context.active_object
        sun.name = "SciGraphs_Light_Sun"
        sun.data.energy = 5.0
        sun.rotation_euler = (0.785, 0, 0.785)  # 45 degrees


def register():
    bpy.utils.register_class(SCIGRAPHS_OT_UpdateAppearance)
    bpy.utils.register_class(SCIGRAPHS_OT_ApplyRenderingPreset)
    bpy.utils.register_class(SCIGRAPHS_OT_SetupLighting)


def unregister():
    bpy.utils.unregister_class(SCIGRAPHS_OT_SetupLighting)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ApplyRenderingPreset)
    bpy.utils.unregister_class(SCIGRAPHS_OT_UpdateAppearance)

