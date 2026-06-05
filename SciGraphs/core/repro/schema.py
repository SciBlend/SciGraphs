# Pipeline schema validation for reproducible SciGraphs workflows
#
# Defines the structure of a declarative pipeline and validates user input.

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import os


class ValidationError(Exception):
    """Raised when pipeline validation fails."""
    pass


class DatasetSource(Enum):
    OSMNX = "osmnx"
    GEXF = "gexf"
    GRAPHML = "graphml"
    CSV = "csv"
    SUITESPARSE = "suitesparse"
    SQL = "sql"
    CITY2GRAPH = "city2graph"


class LayoutAlgorithm(Enum):
    GRID = "GRID"
    SPRING = "SPRING"
    SPRING_3D = "SPRING_3D"
    FORCEATLAS2 = "FORCEATLAS2"
    IGRAPH_DRL_2D = "IGRAPH_DRL_2D"
    IGRAPH_DH = "IGRAPH_DH"
    IGRAPH_GRAPHOPT = "IGRAPH_GRAPHOPT"
    CIRCLE_PACKING = "CIRCLE_PACKING"
    FRUCHTERMAN_REINGOLD = "FRUCHTERMAN_REINGOLD"
    KAMADA_KAWAI = "KAMADA_KAWAI"
    YIFAN_HU = "YIFAN_HU"
    GRAPHVIZ_DOT = "GRAPHVIZ_DOT"
    GRAPHVIZ_NEATO = "GRAPHVIZ_NEATO"
    GRAPHVIZ_FDP = "GRAPHVIZ_FDP"
    GRAPHVIZ_SFDP = "GRAPHVIZ_SFDP"
    GRAPHVIZ_TWOPI = "GRAPHVIZ_TWOPI"
    GRAPHVIZ_CIRCO = "GRAPHVIZ_CIRCO"
    GRAPHVIZ_OSAGE = "GRAPHVIZ_OSAGE"
    GRAPHVIZ_PATCHWORK = "GRAPHVIZ_PATCHWORK"
    IGRAPH_DRL = "IGRAPH_DRL"
    IGRAPH_FR = "IGRAPH_FR"
    IGRAPH_KK = "IGRAPH_KK"
    IGRAPH_LGL = "IGRAPH_LGL"
    SPECTRAL = "SPECTRAL"
    SPECTRAL_3D = "SPECTRAL_3D"
    CIRCULAR = "CIRCULAR"
    SHELL = "SHELL"
    RANDOM = "RANDOM"
    SPHERE = "SPHERE"
    SPIRAL_3D = "SPIRAL_3D"
    HELIX = "HELIX"
    CUBE = "CUBE"
    HIERARCHICAL_3D = "HIERARCHICAL_3D"
    MDS_3D = "MDS_3D"
    BIPARTITE_3D = "BIPARTITE_3D"
    FORCE_ATLAS2 = "FORCE_ATLAS2"
    SUGIYAMA = "SUGIYAMA"
    CIRCULAR_HIERARCHY = "CIRCULAR_HIERARCHY"


class RenderEngine(Enum):
    CYCLES = "CYCLES"
    EEVEE = "BLENDER_EEVEE_NEXT"
    WORKBENCH = "BLENDER_WORKBENCH"


# Schema definitions with defaults and validation rules
SCHEMA = {
    "meta": {
        "type": "object",
        "required": ["title"],
        "properties": {
            "title": {"type": "string", "description": "Pipeline identifier"},
            "seed": {"type": "integer", "default": 42, "description": "Global random seed"},
            "output_dir": {"type": "string", "default": "//repro/default", "description": "Output directory (// = blend file relative)"},
            "description": {"type": "string", "default": "", "description": "Human-readable description"},
            "version": {"type": "string", "default": "1.0", "description": "Pipeline version"},
        }
    },
    "dataset": {
        "type": "object",
        "required": ["source"],
        "properties": {
            "source": {"type": "string", "enum": [e.value for e in DatasetSource]},
            # OSMnx-specific
            "method": {"type": "string", "enum": ["PLACE", "BBOX", "POINT", "ADDRESS", "POLYGON"]},
            "query": {"type": "string"},
            "network_type": {"type": "string", "default": "drive", "enum": ["drive", "walk", "bike", "all", "all_public", "all_private", "drive_service"]},
            "simplify": {"type": "boolean", "default": True},
            "cache": {"type": "boolean", "default": True},
            "retain_all": {"type": "boolean", "default": False},
            # File-based
            "filepath": {"type": "string"},
            # GEXF/GraphML specifics
            "auto_layout": {"type": "boolean", "default": True},
            # SQL-specific
            "connection_string": {"type": "string"},
            "nodes_query": {"type": "string"},
            "edges_query": {"type": "string"},
            # SuiteSparse-specific
            "matrix_name": {"type": "string"},
            # City2Graph-specific
            "bbox": {"type": "array", "items": {"type": "number"}},
            "layers": {"type": "array", "items": {"type": "string"}},
        }
    },
    "analysis": {
        "type": "object",
        "properties": {
            "metrics": {"type": "array", "items": {"type": "string"}, "default": []},
            "clustering": {
                "type": "object",
                "properties": {
                    "algorithm": {"type": "string", "default": "rn", "enum": ["cpm", "infomap", "rb", "rn", "rnsc", "scluster", "uvcluster", "louvain", "leiden"]},
                    "resolution": {"type": "number", "default": 1.0},
                }
            },
            "normalize": {"type": "boolean", "default": True},
        }
    },
    "layout": {
        "type": "object",
        "properties": {
            "algorithm": {"type": "string", "default": "YIFAN_HU", "enum": [e.value for e in LayoutAlgorithm]},
            "scale": {"type": "number", "default": 1.0},
            "iterations": {"type": "integer", "default": 50},
            "seed": {"type": "integer", "description": "Override meta.seed for layout only"},
            "dimension": {"type": "integer", "default": 3, "enum": [2, 3]},
            # Spring-specific
            "k": {"type": "number", "description": "Optimal distance between nodes"},
            # Force Atlas 2-specific
            "gravity": {"type": "number", "default": 1.0},
            "scaling_ratio": {"type": "number", "default": 2.0},
        }
    },
    "visual": {
        "type": "object",
        "properties": {
            "setup_geometry_nodes": {"type": "boolean", "default": True},
            "node_color": {"type": "string", "description": "Attribute name for node coloring"},
            "edge_color": {"type": "string", "description": "Attribute name for edge coloring"},
            "node_size": {"type": "string", "description": "Attribute name for node sizing"},
            "edge_width": {"type": "string", "description": "Attribute name for edge width"},
            "node_min_size": {"type": "number", "default": 0.01},
            "node_max_size": {"type": "number", "default": 0.1},
            "edge_min_width": {"type": "number", "default": 0.002},
            "edge_max_width": {"type": "number", "default": 0.02},
            "colormap": {"type": "string", "default": "viridis"},
            "rendering_preset": {"type": "string", "enum": ["SCIENTIFIC", "PRESENTATION", "PRINT", "CUSTOM"]},
            "edge_style": {"type": "string", "enum": ["GEPHI_DEFAULT", "CYTOSCAPE_BEZIER", "YFILES_ORGANIC", "GRAPHVIZ_SPLINE", "TULIP_CURVED", "CURVED_UNIFORM"]},
        }
    },
    "render": {
        "type": "object",
        "properties": {
            "engine": {"type": "string", "default": "CYCLES", "enum": [e.value for e in RenderEngine]},
            "resolution": {"type": "array", "items": {"type": "integer"}, "default": [1920, 1080]},
            "samples": {"type": "integer", "default": 128},
            "camera": {"type": "string", "description": "Camera object name"},
            "output": {"type": "string", "default": "render.png", "description": "Output filename"},
            "transparent": {"type": "boolean", "default": False},
            "denoise": {"type": "boolean", "default": True},
        }
    },
    "exports": {
        "type": "object",
        "properties": {
            "graph": {"type": "string", "description": "Export graph as GEXF/GraphML"},
            "positions": {"type": "string", "description": "Export node positions CSV"},
            "statistics": {"type": "string", "description": "Export statistics report"},
            "blend": {"type": "string", "description": "Save .blend file copy"},
        }
    },
    "ops": {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["id"],
            "properties": {
                "id": {"type": "string", "description": "Operator bl_idname (e.g., scigraphs.apply_layout) or a registry shortcut"},
                "props": {"type": "object", "description": "Operator keyword properties passed directly to the operator call"},
                "scene_props": {
                    "type": "object",
                    "description": (
                        "Scene properties to set before calling. Either a flat "
                        "mapping applied to scene.scigraphs, or a mapping keyed "
                        "by property group: scigraphs, city2graph, coloring, "
                        "viz, repro, splitter. Example: "
                        "{\"city2graph\": {\"prox_knn_k\": 8}, \"coloring\": {\"colormap\": \"magma\"}}"
                    ),
                },
            }
        },
        "default": []
    }
}


@dataclass
class MetaSpec:
    title: str
    seed: int = 42
    output_dir: str = "//repro/default"
    description: str = ""
    version: str = "1.0"


@dataclass
class DatasetSpec:
    source: str
    method: Optional[str] = None
    query: Optional[str] = None
    network_type: str = "drive"
    simplify: bool = True
    cache: bool = True
    retain_all: bool = False
    filepath: Optional[str] = None
    auto_layout: bool = True
    connection_string: Optional[str] = None
    nodes_query: Optional[str] = None
    edges_query: Optional[str] = None
    matrix_name: Optional[str] = None
    bbox: Optional[List[float]] = None
    layers: Optional[List[str]] = None


@dataclass
class ClusteringSpec:
    algorithm: str = "rn"
    resolution: float = 1.0


@dataclass
class AnalysisSpec:
    metrics: List[str] = field(default_factory=list)
    clustering: Optional[ClusteringSpec] = None
    normalize: bool = True


@dataclass
class LayoutSpec:
    algorithm: str = "YIFAN_HU"
    scale: float = 1.0
    iterations: int = 50
    seed: Optional[int] = None
    dimension: int = 3
    k: Optional[float] = None
    gravity: float = 1.0
    scaling_ratio: float = 2.0


@dataclass
class VisualSpec:
    setup_geometry_nodes: bool = True
    node_color: Optional[str] = None
    edge_color: Optional[str] = None
    node_size: Optional[str] = None
    edge_width: Optional[str] = None
    node_min_size: float = 0.01
    node_max_size: float = 0.1
    edge_min_width: float = 0.002
    edge_max_width: float = 0.02
    colormap: str = "viridis"
    rendering_preset: Optional[str] = None
    edge_style: Optional[str] = None


@dataclass
class RenderSpec:
    engine: str = "CYCLES"
    resolution: List[int] = field(default_factory=lambda: [1920, 1080])
    samples: int = 128
    camera: Optional[str] = None
    output: str = "render.png"
    transparent: bool = False
    denoise: bool = True


@dataclass
class ExportsSpec:
    graph: Optional[str] = None
    positions: Optional[str] = None
    statistics: Optional[str] = None
    blend: Optional[str] = None


@dataclass
class OpSpec:
    id: str
    props: Dict[str, Any] = field(default_factory=dict)
    scene_props: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineSchema:
    """Complete pipeline specification."""
    meta: MetaSpec
    dataset: Optional[DatasetSpec] = None
    analysis: Optional[AnalysisSpec] = None
    layout: Optional[LayoutSpec] = None
    visual: Optional[VisualSpec] = None
    render: Optional[RenderSpec] = None
    exports: Optional[ExportsSpec] = None
    ops: List[OpSpec] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        from dataclasses import asdict
        result = {}
        result["meta"] = asdict(self.meta)
        if self.dataset:
            result["dataset"] = {k: v for k, v in asdict(self.dataset).items() if v is not None}
        if self.analysis:
            d = asdict(self.analysis)
            if d.get("clustering") is None:
                del d["clustering"]
            result["analysis"] = d
        if self.layout:
            result["layout"] = {k: v for k, v in asdict(self.layout).items() if v is not None}
        if self.visual:
            result["visual"] = {k: v for k, v in asdict(self.visual).items() if v is not None}
        if self.render:
            result["render"] = asdict(self.render)
        if self.exports:
            result["exports"] = {k: v for k, v in asdict(self.exports).items() if v is not None}
        if self.ops:
            result["ops"] = [asdict(op) for op in self.ops]
        return result


def _apply_defaults(data: Dict[str, Any], schema_section: Dict) -> Dict[str, Any]:
    """Apply default values from schema to data."""
    result = dict(data)
    props = schema_section.get("properties", {})
    for key, prop_def in props.items():
        if key not in result and "default" in prop_def:
            result[key] = prop_def["default"]
    return result


def _validate_type(value: Any, expected_type: str, path: str) -> None:
    """Validate a value against an expected type."""
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    expected = type_map.get(expected_type)
    if expected and not isinstance(value, expected):
        raise ValidationError(f"{path}: expected {expected_type}, got {type(value).__name__}")


def _validate_enum(value: Any, enum_values: List[str], path: str) -> None:
    """Validate a value is one of the allowed enum values."""
    if value not in enum_values:
        raise ValidationError(f"{path}: '{value}' not in {enum_values}")


def _validate_section(data: Dict[str, Any], schema_section: Dict, path: str) -> None:
    """Validate a section of the pipeline against its schema."""
    props = schema_section.get("properties", {})
    required = schema_section.get("required", [])

    for req in required:
        if req not in data:
            raise ValidationError(f"{path}: missing required field '{req}'")

    for key, value in data.items():
        if key not in props:
            continue  # Allow extra fields for extensibility

        prop_def = props[key]
        field_path = f"{path}.{key}"

        if value is None:
            continue

        _validate_type(value, prop_def.get("type", "string"), field_path)

        if "enum" in prop_def:
            _validate_enum(value, prop_def["enum"], field_path)

        if prop_def.get("type") == "object" and "properties" in prop_def:
            _validate_section(value, prop_def, field_path)


def validate_pipeline(data: Dict[str, Any]) -> List[str]:
    """
    Validate a pipeline dictionary against the schema.

    Args:
        data: Pipeline dictionary (parsed from YAML/JSON)

    Returns:
        List of warning messages (empty if no warnings)

    Raises:
        ValidationError: If validation fails
    """
    warnings = []

    if not isinstance(data, dict):
        raise ValidationError("Pipeline must be a dictionary/object")

    # Validate meta (required)
    if "meta" not in data:
        raise ValidationError("Pipeline must have a 'meta' section")
    _validate_section(data["meta"], SCHEMA["meta"], "meta")

    # Validate optional sections
    for section in ["dataset", "analysis", "layout", "visual", "render", "exports"]:
        if section in data and data[section] is not None:
            _validate_section(data[section], SCHEMA[section], section)

    # Validate ops array
    if "ops" in data and data["ops"]:
        if not isinstance(data["ops"], list):
            raise ValidationError("'ops' must be an array")
        for i, op in enumerate(data["ops"]):
            if not isinstance(op, dict):
                raise ValidationError(f"ops[{i}]: must be an object")
            if "id" not in op:
                raise ValidationError(f"ops[{i}]: missing required field 'id'")

    # Check for potential issues
    if "dataset" not in data and not data.get("ops"):
        warnings.append("No dataset or ops specified - pipeline may not do anything")

    if "render" in data and "camera" not in data.get("render", {}):
        warnings.append("Render specified without camera - will use active camera")

    return warnings


def get_default_pipeline(title: str = "untitled") -> Dict[str, Any]:
    """Get a minimal default pipeline structure."""
    return {
        "meta": {
            "title": title,
            "seed": 42,
            "output_dir": "//repro/default",
        }
    }


def _dataclass_kwargs(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    """Filter a dictionary to keyword arguments accepted by a dataclass."""
    field_names = {item.name for item in fields(cls)}
    return {key: value for key, value in data.items() if key in field_names}


def parse_spec(data: Dict[str, Any]) -> PipelineSchema:
    """
    Parse a validated pipeline dictionary into typed dataclasses.

    Args:
        data: Validated pipeline dictionary

    Returns:
        PipelineSchema instance
    """
    # Apply defaults and parse meta
    meta_data = _apply_defaults(data.get("meta", {}), SCHEMA["meta"])
    meta = MetaSpec(**meta_data)

    # Parse dataset
    dataset = None
    if "dataset" in data and data["dataset"]:
        ds_data = _apply_defaults(data["dataset"], SCHEMA["dataset"])
        dataset = DatasetSpec(**_dataclass_kwargs(DatasetSpec, ds_data))

    # Parse analysis
    analysis = None
    if "analysis" in data and data["analysis"]:
        an_data = _apply_defaults(data["analysis"], SCHEMA["analysis"])
        clustering = None
        if "clustering" in an_data and an_data["clustering"]:
            cl_data = _apply_defaults(an_data["clustering"], SCHEMA["analysis"]["properties"]["clustering"])
            clustering = ClusteringSpec(**cl_data)
        analysis = AnalysisSpec(
            metrics=an_data.get("metrics", []),
            clustering=clustering,
            normalize=an_data.get("normalize", True),
        )

    # Parse layout
    layout = None
    if "layout" in data and data["layout"]:
        ly_data = _apply_defaults(data["layout"], SCHEMA["layout"])
        layout = LayoutSpec(**_dataclass_kwargs(LayoutSpec, ly_data))

    # Parse visual
    visual = None
    if "visual" in data and data["visual"]:
        vs_data = _apply_defaults(data["visual"], SCHEMA["visual"])
        visual = VisualSpec(**_dataclass_kwargs(VisualSpec, vs_data))

    # Parse render
    render = None
    if "render" in data and data["render"]:
        rn_data = _apply_defaults(data["render"], SCHEMA["render"])
        render = RenderSpec(**_dataclass_kwargs(RenderSpec, rn_data))

    # Parse exports
    exports = None
    if "exports" in data and data["exports"]:
        ex_data = data["exports"]
        exports = ExportsSpec(**_dataclass_kwargs(ExportsSpec, ex_data))

    # Parse ops
    ops = []
    if "ops" in data and data["ops"]:
        for op_data in data["ops"]:
            ops.append(OpSpec(
                id=op_data["id"],
                props=op_data.get("props", {}),
                scene_props=op_data.get("scene_props", {}),
            ))

    return PipelineSchema(
        meta=meta,
        dataset=dataset,
        analysis=analysis,
        layout=layout,
        visual=visual,
        render=render,
        exports=exports,
        ops=ops,
    )
