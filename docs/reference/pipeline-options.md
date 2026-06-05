# SciGraphs pipeline options reference

Auto-generated reference of every option exposed to reproducible pipelines. Regenerate it from the Reproducibility panel (Export Options Reference) or via `SciGraphs.core.repro.reference.generate_reference_markdown()`.

## Declarative stages

These typed sections cover the common workflow. Every field below is applied by the executor.

### `meta`

| Field | Type | Default | Enum |
| --- | --- | --- | --- |
| `title` | string |  |  |
| `seed` | integer | 42 |  |
| `output_dir` | string | //repro/default |  |
| `description` | string |  |  |
| `version` | string | 1.0 |  |

### `dataset`

| Field | Type | Default | Enum |
| --- | --- | --- | --- |
| `source` | string |  | osmnx, gexf, graphml, csv, suitesparse, sql, city2graph |
| `method` | string |  | PLACE, BBOX, POINT, ADDRESS, POLYGON |
| `query` | string |  |  |
| `network_type` | string | drive | drive, walk, bike, all, all_public, all_private, drive_service |
| `simplify` | boolean | True |  |
| `cache` | boolean | True |  |
| `retain_all` | boolean | False |  |
| `filepath` | string |  |  |
| `auto_layout` | boolean | True |  |
| `connection_string` | string |  |  |
| `nodes_query` | string |  |  |
| `edges_query` | string |  |  |
| `matrix_name` | string |  |  |
| `bbox` | array |  |  |
| `layers` | array |  |  |

### `analysis`

| Field | Type | Default | Enum |
| --- | --- | --- | --- |
| `metrics` | array | [] |  |
| `clustering` | object |  |  |
| `normalize` | boolean | True |  |

#### `analysis.clustering`

| Field | Type | Default | Enum |
| --- | --- | --- | --- |
| `algorithm` | string | rn | cpm, infomap, rb, rn, rnsc, scluster, uvcluster, louvain, leiden |
| `resolution` | number | 1.0 |  |

### `layout`

| Field | Type | Default | Enum |
| --- | --- | --- | --- |
| `algorithm` | string | YIFAN_HU | GRID, SPRING, SPRING_3D, FORCEATLAS2, IGRAPH_DRL_2D, IGRAPH_DH, IGRAPH_GRAPHOPT, CIRCLE_PACKING, FRUCHTERMAN_REINGOLD, KAMADA_KAWAI, YIFAN_HU, GRAPHVIZ_DOT, GRAPHVIZ_NEATO, GRAPHVIZ_FDP, GRAPHVIZ_SFDP, GRAPHVIZ_TWOPI, GRAPHVIZ_CIRCO, GRAPHVIZ_OSAGE, GRAPHVIZ_PATCHWORK, IGRAPH_DRL, IGRAPH_FR, IGRAPH_KK, IGRAPH_LGL, SPECTRAL, SPECTRAL_3D, CIRCULAR, SHELL, RANDOM, SPHERE, SPIRAL_3D, HELIX, CUBE, HIERARCHICAL_3D, MDS_3D, BIPARTITE_3D, FORCE_ATLAS2, SUGIYAMA, CIRCULAR_HIERARCHY |
| `scale` | number | 1.0 |  |
| `iterations` | integer | 50 |  |
| `seed` | integer |  |  |
| `dimension` | integer | 3 | 2, 3 |
| `k` | number |  |  |
| `gravity` | number | 1.0 |  |
| `scaling_ratio` | number | 2.0 |  |

### `visual`

| Field | Type | Default | Enum |
| --- | --- | --- | --- |
| `setup_geometry_nodes` | boolean | True |  |
| `node_color` | string |  |  |
| `edge_color` | string |  |  |
| `node_size` | string |  |  |
| `edge_width` | string |  |  |
| `node_min_size` | number | 0.01 |  |
| `node_max_size` | number | 0.1 |  |
| `edge_min_width` | number | 0.002 |  |
| `edge_max_width` | number | 0.02 |  |
| `colormap` | string | viridis |  |
| `rendering_preset` | string |  | SCIENTIFIC, PRESENTATION, PRINT, CUSTOM |
| `edge_style` | string |  | GEPHI_DEFAULT, CYTOSCAPE_BEZIER, YFILES_ORGANIC, GRAPHVIZ_SPLINE, TULIP_CURVED, CURVED_UNIFORM |

### `render`

| Field | Type | Default | Enum |
| --- | --- | --- | --- |
| `engine` | string | CYCLES | CYCLES, BLENDER_EEVEE_NEXT, BLENDER_WORKBENCH |
| `resolution` | array | [1920, 1080] |  |
| `samples` | integer | 128 |  |
| `camera` | string |  |  |
| `output` | string | render.png |  |
| `transparent` | boolean | False |  |
| `denoise` | boolean | True |  |

### `exports`

| Field | Type | Default | Enum |
| --- | --- | --- | --- |
| `graph` | string |  |  |
| `positions` | string |  |  |
| `statistics` | string |  |  |
| `blend` | string |  |  |

## Generic ops

The `ops` array calls any operator by `id` (a bl_idname or a registry shortcut), with `props` (operator keyword arguments) and `scene_props` (scene state). `scene_props` may be flat (applied to `scene.scigraphs`) or keyed by property group.

### Property-group keys for `scene_props`

| Group key | Scene attribute |
| --- | --- |
| `scigraphs` | `scene.scigraphs` |
| `city2graph` | `scene.city2graph` |
| `coloring` | `scene.scigraphs_coloring` |
| `viz` | `scene.scigraphs_viz` |
| `repro` | `scene.scigraphs_repro` |
| `splitter` | `scene.scigraphs_splitter` |

### Registry shortcuts

| Shortcut | Operator |
| --- | --- |
| `appearance` | `scigraphs.update_appearance` |
| `c2g_place` | `scigraphs.c2g_load_overture_place` |
| `centrality` | `scigraphs.calculate_centrality` |
| `clustering` | `scigraphs.apply_clustering` |
| `communities` | `scigraphs.apply_clustering` |
| `consolidate` | `scigraphs.osmnx_consolidate` |
| `create_graph` | `scigraphs.create_graph` |
| `crossings` | `scigraphs.validate_crossings` |
| `dual` | `scigraphs.create_dual_graph` |
| `edge_style` | `scigraphs.apply_edge_style_preset` |
| `elevation_3d` | `scigraphs.osmnx_apply_elevation_3d` |
| `elevation_raster` | `scigraphs.osmnx_add_elevations_raster` |
| `export` | `scigraphs.export_graph` |
| `export_gexf` | `scigraphs.export_gexf` |
| `export_graphml` | `scigraphs.export_graphml` |
| `export_positions` | `scigraphs.export_positions` |
| `export_stats` | `scigraphs.generate_statistics_report` |
| `faces` | `scigraphs.compute_faces` |
| `flow` | `scigraphs.analyze_flow` |
| `genus` | `scigraphs.calculate_genus` |
| `import_osmnx` | `scigraphs.import_osm_graph` |
| `isochrones` | `scigraphs.osmnx_isochrones` |
| `layout` | `scigraphs.apply_layout` |
| `layout_step` | `scigraphs.execute_layout_step` |
| `lighting` | `scigraphs.setup_lighting` |
| `orientation` | `scigraphs.osmnx_orientation_entropy` |
| `osmnx` | `scigraphs.import_osm_graph` |
| `osmnx_centrality` | `scigraphs.osmnx_centrality` |
| `osmnx_export` | `scigraphs.osmnx_export` |
| `osmnx_stats` | `scigraphs.osmnx_basic_stats` |
| `overture` | `scigraphs.c2g_load_overture` |
| `patterns` | `scigraphs.detect_patterns` |
| `planarity` | `scigraphs.check_planarity` |
| `preset` | `scigraphs.apply_rendering_preset` |
| `project` | `scigraphs.osmnx_project_graph` |
| `render` | `render.render` |
| `reset_layout` | `scigraphs.reset_layout` |
| `sccs` | `scigraphs.find_sccs` |
| `setup_vis` | `scigraphs.setup_visualization` |
| `shortest_path` | `scigraphs.osmnx_shortest_path` |
| `simplify` | `scigraphs.osmnx_simplify` |
| `splitter` | `scigraphs.network_splitter_3d` |
| `sql` | `scigraphs.create_graph_from_sql` |
| `stats` | `scigraphs.calculate_global_statistics` |
| `suitesparse` | `scigraphs.download_suitesparse` |
| `surface` | `scigraphs.visualize_surface` |
| `text_overlay` | `scigraphs.generate_text_overlay` |

## Scene property groups

_Property-group tables require a running Blender session. Run the generator from inside Blender to include them._
