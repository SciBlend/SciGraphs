"""Shared feature-tag presets for OSMnx and City2Graph imports."""

FEATURE_TAG_PRESETS = {
    'BUILDING': {"building": True},
    'AMENITY': {"amenity": True},
    'RESTAURANT': {"amenity": ["restaurant"]},
    'SHOP': {"shop": True},
    'LEISURE': {"leisure": True},
    'PARKING': {"amenity": ["parking"]},
    'BUS_STOP': {"highway": ["bus_stop"]},
    'RAIL_STATION': {"railway": ["station", "halt", "stop", "tram_stop"]},
    'PARK': {"leisure": ["park", "garden", "nature_reserve", "playground"]},
    'EDUCATION': {"amenity": ["school", "university", "college", "kindergarten", "library"]},
    'HEALTH': {"amenity": ["hospital", "clinic", "doctors", "pharmacy", "dentist"]},
    'AMENITY_METAPATH': {"amenity": ["cafe", "restaurant", "pub", "bar", "museum", "theatre", "cinema"]},
    'LANDUSE': {"landuse": True},
    'NATURAL': {"natural": True},
    'WATER': {"natural": ["water"]},
    'HIGHWAY': {"highway": True},
}

FEATURE_SOURCE_ITEMS = [
    ('OVERTURE', "Overture / City2Graph", "Use Overture where available and OSMnx fallbacks otherwise"),
    ('OSMNX', "OSMnx / Overpass", "Use OSMnx feature queries with OSM tags"),
]

FEATURE_TYPE_ITEMS = [
    (key, key.replace('_', ' ').title(), "")
    for key in FEATURE_TAG_PRESETS
] + [('CUSTOM', "Custom Tags", "Use custom OSM key=value tags")]


def feature_type_items_for_source(source):
    """Return the feature-type enum items supported by a given source.

    OSMnx supports every preset plus custom tags. Overture only exposes the
    presets that map to one of its native layers (building, places, segment,
    water, land); presets without an Overture mapping, and custom tags, are
    omitted so the list never offers an option the source cannot serve.
    """
    if source == 'OSMNX':
        return list(FEATURE_TYPE_ITEMS)

    return [
        (key, label, desc)
        for (key, label, desc) in FEATURE_TYPE_ITEMS
        if key in OVERTURE_PRESET_TYPES
    ]

OVERTURE_PRESET_TYPES = {
    'BUILDING': 'building',
    'AMENITY': 'place',
    'RESTAURANT': 'place',
    'SHOP': 'place',
    'LEISURE': 'place',
    'PARKING': 'place',
    'BUS_STOP': 'place',
    'RAIL_STATION': 'place',
    'PARK': 'land',
    'EDUCATION': 'place',
    'HEALTH': 'place',
    'AMENITY_METAPATH': 'place',
    'LANDUSE': 'land',
    'NATURAL': 'land',
    'WATER': 'water',
    'HIGHWAY': 'segment',
}

OVERTURE_PLACE_CATEGORY_KEYWORDS = {
    'RESTAURANT': ['restaurant', 'fast_food', 'food'],
    'SHOP': ['shop', 'store', 'retail', 'shopping', 'market'],
    'LEISURE': ['leisure', 'park', 'sports', 'fitness', 'entertainment', 'recreation'],
    'PARKING': ['parking'],
    'BUS_STOP': ['bus', 'transit', 'transport'],
    'RAIL_STATION': ['rail', 'train', 'station', 'subway', 'metro', 'tram'],
    'EDUCATION': ['school', 'university', 'college', 'kindergarten', 'library', 'education'],
    'HEALTH': ['hospital', 'clinic', 'doctor', 'pharmacy', 'dentist', 'health', 'medical'],
    'AMENITY_METAPATH': ['cafe', 'restaurant', 'pub', 'bar', 'museum', 'theatre', 'theater', 'cinema'],
}


def tags_from_preset(preset, custom_tags_str=""):
    """Resolve a preset enum or custom key=value string to an OSM tags dict."""
    if preset != 'CUSTOM':
        return dict(FEATURE_TAG_PRESETS.get(preset, {"building": True}))

    tags = {}
    for pair in custom_tags_str.split(','):
        if '=' not in pair:
            continue
        key, value = pair.split('=', 1)
        key = key.strip()
        value = value.strip()
        if key:
            tags[key] = True if value.lower() in {"true", "yes", "*"} else value
    return tags


def resolve_feature_tags(props):
    """Resolve shared scene feature-import properties to an OSM tags dict."""
    return tags_from_preset(
        getattr(props, "feat_type", "BUILDING"),
        getattr(props, "feat_custom_tags", ""),
    )


def overture_type_from_preset(preset):
    """Return the closest City2Graph/Overture feature type for a preset."""
    return OVERTURE_PRESET_TYPES.get(preset)


def overture_place_keywords(preset):
    """Return category keywords used to filter Overture places for a preset.

    Returns ``None`` for presets that should not be filtered (e.g. the broad
    ``AMENITY`` preset, which maps to all places).
    """
    return OVERTURE_PLACE_CATEGORY_KEYWORDS.get(preset)
