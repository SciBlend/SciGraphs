"""The interactive SciGraphs_Viz Geometry Nodes tree, built with nodebpy.

Glyphs are instanced on the graph's vertices, tubes are swept along its edges
and arrow cones are placed along directed edges, all driven by the modifier's
input sockets so the UI panel can change them without a rebuild.
"""

from nodebpy import TreeBuilder
from nodebpy import geometry as g

from .geometry import (
    _BUILTIN_ATTRS,
    _MESH_TYPE_TO_GN,
    EDGE_DIRECTION_ATTRIBUTE,
    EDGE_SCALE_ATTRIBUTE,
    EDGE_SPAN_ATTRIBUTES,
    FILTER_PASS_ATTRIBUTE,
)

TREE_NAME = "SciGraphs_Interactive_Viz"

# Frame colours of the four blocks of the tree.
_FRAME_COLOURS = {
    "Shared": (0.28, 0.26, 0.32),
    "Nodes": (0.22, 0.34, 0.28),
    "Edges": (0.20, 0.28, 0.38),
    "Arrows": (0.36, 0.28, 0.20),
}


def _frame(label):
    frame = g.Frame(label)
    frame.node.use_custom_color = True
    frame.node.color = _FRAME_COLOURS[label]
    return frame


def _named(node, name, label=None):
    """Give *node* a fixed name: the visualization operators look nodes up by it."""
    node.node.name = name
    if label is not None:
        node.node.label = label
    return node


def _promote_edge_attributes(geometry, mesh):
    """Re-domain EDGE attributes to POINT so they survive Mesh to Curve."""
    skip = ({'.edge_verts', 'sharp_edge', 'crease_edge', 'material_index',
             EDGE_DIRECTION_ATTRIBUTE} | set(EDGE_SPAN_ATTRIBUTES))
    edge_attrs = [(a.name, a.data_type) for a in mesh.attributes
                  if a.domain == 'EDGE' and a.name not in skip
                  and not a.name.startswith('.')]
    # The edge scale socket reads this name even before the mesh has it.
    if EDGE_SCALE_ATTRIBUTE not in mesh.attributes:
        edge_attrs.append((EDGE_SCALE_ATTRIBUTE, 'FLOAT'))

    for name, mesh_type in edge_attrs:
        gn_type = _MESH_TYPE_TO_GN.get(mesh_type)
        if gn_type is None:
            continue
        value = g.NamedAttribute(name, data_type=gn_type).o.attribute
        geometry = g.StoreNamedAttribute(geometry, name=name, value=value,
                                         data_type=gn_type, domain='POINT')
    return geometry


def _strip_custom_attributes(geometry, obj):
    """Remove the graph's custom attributes, one Remove Named Attribute each.

    Without this, node values bleed into the edge tubes through Mesh to Curve.
    The coloring code splices out the node for the attribute it colors by, so
    each attribute gets its own node rather than a wildcard pattern."""
    color_attr = obj.get("scigraphs_color_attr", "")
    keep = {color_attr, f"{color_attr}_color"} if color_attr else set()
    for attr in obj.data.attributes:
        if (attr.name in _BUILTIN_ATTRS or attr.name.startswith('.')
                or attr.name in keep):
            continue
        geometry = g.RemoveNamedAttribute(geometry, name=attr.name)
    return geometry


def _scale_socket(attr_name, multiplier, base):
    """*base* times the named attribute and *multiplier*; just *base* when the
    attribute name is empty."""
    scaled = g.NamedAttribute.float(attr_name).o.attribute * multiplier
    return (attr_name.length() > 0).switch.float(1.0, scaled) * base


def build_interactive_tree(obj, node_group, material=None):
    """Rebuild *node_group* as the interactive tree for *obj*'s mesh.

    The tree depends on the mesh as well as its sockets: which edge attributes
    are promoted and stripped, and whether OSMnx curve points are hidden."""
    mesh = obj.data
    has_intersections = (obj.get("is_osmnx", False)
                         or "is_intersection" in mesh.attributes)

    # Cleared here rather than with clear=True: SciGraphs patches this tree
    # after building it (coloring, the Apply operator), and nodebpy would keep
    # each patched tree as a "(edited)" copy on the next rebuild.
    node_group.nodes.clear()
    node_group.interface.clear()

    with TreeBuilder(node_group) as tree:
        geometry = tree.inputs.geometry("Geometry")
        node_scale = tree.inputs.float("Node Scale", 0.1, min_value=0.001, max_value=5.0)
        node_resolution = tree.inputs.integer("Node Resolution", 3, min_value=1, max_value=6)
        node_shape = tree.inputs.integer("Node Shape", 0)
        node_attr_name = tree.inputs.string("Node Attr Name")
        node_attr_mult = tree.inputs.float("Node Attr Mult", 1.0)
        edge_thickness = tree.inputs.float("Edge Thickness", 0.02, min_value=0.001, max_value=1.0)
        edge_resolution = tree.inputs.integer("Edge Resolution", 4, min_value=3, max_value=32)
        edge_attr_name = tree.inputs.string("Edge Attr Name")
        edge_attr_mult = tree.inputs.float("Edge Attr Mult", 1.0)
        show_arrows = tree.inputs.boolean("Show Arrows", False)
        arrow_size = tree.inputs.float("Arrow Size", 0.15)
        arrow_resolution = tree.inputs.integer("Arrow Resolution", 8, min_value=3, max_value=64)
        arrow_sharpness = tree.inputs.float("Arrow Sharpness", 2.5, min_value=0.5, max_value=12.0)
        arrow_position = tree.inputs.float("Arrow Position", 0.7, min_value=0.0, max_value=1.0)
        filter_enable = tree.inputs.boolean("Filter Enable", False)
        filter_attr = tree.inputs.string("Filter Attr")
        filter_min = tree.inputs.float("Filter Min", 0.0)
        filter_max = tree.inputs.float("Filter Max", 1.0)
        output = tree.outputs.geometry("Geometry")

        with _frame("Shared"):
            prepared = _promote_edge_attributes(geometry, mesh)

            # Filter Enable off passes everything; on, the attribute's range.
            filtered = g.NamedAttribute.float(filter_attr).o.attribute
            in_range = (filtered >= filter_min) & (filtered <= filter_max)
            passes = filter_enable.switch.boolean(True, in_range)

            # Edges are kept when both of their vertices pass: the point value
            # is interpolated onto the edge, so anything below 1 has a
            # filtered end.
            stored_pass = g.StoreNamedAttribute.point.float(
                prepared, name=FILTER_PASS_ATTRIBUTE, value=passes)
            edge_geometry = g.DeleteGeometry.edge(
                stored_pass,
                g.NamedAttribute.float(FILTER_PASS_ATTRIBUTE).o.attribute < 0.999)

        with _frame("Nodes"):
            shapes = (
                _named(g.UVSphere(segments=node_resolution, rings=node_resolution),
                       "SciGraphs_NodeSphere"),
                _named(g.IcoSphere(subdivisions=node_resolution), "SciGraphs_NodeIcoSphere"),
                _named(g.Cube(), "SciGraphs_NodeCube"),
                _named(g.Cone(vertices=node_resolution, depth=1.5), "SciGraphs_NodeCone"),
                _named(g.Cylinder(vertices=node_resolution, depth=1.5), "SciGraphs_NodeCylinder"),
            )
            glyph = (
                _named(g.IndexSwitch.geometry(node_shape, shapes), "SciGraphs_NodeShapeSwitch")
                # Smooth by Angle is an Essentials asset, not a built-in node;
                # the name is what the operators check for.
                >> _named(g.SetShadeSmooth.face(), "SciGraphs_NodeSmoothByAngle",
                          "Smooth by Angle")
            )

            selection = passes
            if has_intersections:
                # OSMnx meshes carry curve points between intersections; only
                # the intersections are graph nodes.
                selection = selection & (
                    g.NamedAttribute.integer("is_intersection").o.attribute == 1)

            node_geometry = (
                g.MeshToPoints(prepared)
                >> g.InstanceOnPoints(
                    selection=selection, instance=glyph,
                    scale=_scale_socket(node_attr_name, node_attr_mult, node_scale))
                >> g.RealizeInstances()
            )

        with _frame("Edges"):
            curves = edge_geometry
            if not has_intersections:
                # Mesh to Curve joins connected edges into one spline, which
                # would run each tube through the nodes it passes. OSMnx
                # meshes are left whole: their splines already break at
                # intersections, and splitting would open gaps in the streets.
                curves = g.SplitEdges(curves)
            tube_geometry = (
                g.MeshToCurve(curves)
                >> g.SetCurveRadius(radius=1.0)
                >> g.CurveToMesh(
                    profile_curve=_named(g.CurveCircle(edge_resolution),
                                         "SciGraphs_EdgeProfile"),
                    scale=_scale_socket(edge_attr_name, edge_attr_mult, edge_thickness))
            )
            tube_geometry = _strip_custom_attributes(tube_geometry, obj)

        with _frame("Arrows"):
            ends = g.EdgeVertices()
            direction = ends.o.position_2 - ends.o.position_1

            # Edges split into several mesh edges (OSMnx curve points) carry
            # the fraction of the whole edge they span; the arrow goes on the
            # one piece that contains Arrow Position.
            u0 = g.NamedAttribute.float(EDGE_SPAN_ATTRIBUTES[0]).o.attribute
            span_end = g.NamedAttribute.float(EDGE_SPAN_ATTRIBUTES[1])
            u1 = span_end.o.attribute
            spans_ok = span_end.o.exists & (u1 > u0)
            position = arrow_position.clamp(max=0.999999)
            in_span = (position >= u0) & (position < u1)
            local = g.Math.divide(position - u0, u1 - u0, use_clamp=True)
            along = direction.mul_add(spans_ok.switch.float(arrow_position, local),
                                      ends.o.position_1)

            # Without spans, one arrow per edge: on the piece ending at a node.
            is_node = g.NamedAttribute.integer("is_intersection")
            ends_at_node = g.SampleIndex.point.integer(
                edge_geometry, is_node.o.attribute, ends.o.vertex_index_2) == 1
            one_per_edge = is_node.o.exists.switch.boolean(True, ends_at_node)
            carries_arrow = spans_ok.switch.boolean(one_per_edge, in_span)

            cone = g.Cone(vertices=arrow_resolution, radius_bottom=arrow_size,
                          depth=arrow_size * arrow_sharpness)
            heading = g.AlignRotationToVector(
                vector=g.NamedAttribute.vector(EDGE_DIRECTION_ATTRIBUTE).o.attribute)
            arrow_geometry = (
                g.StoreNamedAttribute.edge.vector(
                    edge_geometry, name=EDGE_DIRECTION_ATTRIBUTE, value=direction)
                >> g.MeshToPoints.edges(selection=carries_arrow, position=along)
                >> g.InstanceOnPoints(selection=show_arrows, instance=cone,
                                      rotation=heading)
                >> g.RealizeInstances()
            )
            arrow_geometry = _strip_custom_attributes(arrow_geometry, obj)

        (
            g.JoinGeometry((node_geometry, tube_geometry, arrow_geometry))
            >> g.SetMaterial(material=material)
            >> output
        )

    return node_group
