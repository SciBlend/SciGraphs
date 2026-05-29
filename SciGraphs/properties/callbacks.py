"""Dynamic enum callbacks shared across property modules."""


def get_column_items(self, context):
    """Dynamic enum callback for column selection."""
    items = []

    if not hasattr(context.scene, 'scigraphs'):
        items.append(('0', 'None', 'No columns available'))
        return items

    props = context.scene.scigraphs

    if props.data_source == 'DATABASE':
        if props.sql_columns_cache:
            columns = props.sql_columns_cache.split('|')
            for i, col in enumerate(columns):
                if col:
                    items.append((str(i), col, f"Column {i}: {col}"))
    else:
        if props.filepath:
            from ..core import importer
            columns = importer.get_columns_from_file(props.filepath, props.csv_delimiter)

            for i, col in enumerate(columns):
                items.append((str(i), col, f"Column {i}: {col}"))

    if not items:
        items.append(('0', 'None', 'No columns available'))

    return items


def get_db_profile_items(self, context):
    """Dynamic enum callback for database profile selection."""
    items = []

    from ..preferences import get_preferences
    prefs = get_preferences()

    if prefs and prefs.db_profiles:
        for i, profile in enumerate(prefs.db_profiles):
            items.append((str(i), profile.name, f"{profile.db_type}: {profile.name}"))

    if not items:
        items.append(('-1', '(No profiles)', 'Configure profiles in Preferences'))

    return items


def get_attribute_items(self, context):
    """Dynamic enum callback for available attributes from the active graph object."""
    items = []

    obj = context.active_object
    if obj is not None and obj.type == 'MESH':
        from ..core import text_overlay
        attributes = text_overlay.get_available_attributes(obj)

        for attr in attributes:
            items.append((attr, attr, f"Attribute: {attr}"))

    if not items:
        items.append(('NONE', '(No attributes)', 'No attributes available'))

    return items


def get_system_fonts(self, context):
    """Get available system fonts."""
    import os
    items = []

    font_dirs = []

    font_dirs.extend([
        '/usr/share/fonts',
        '/usr/local/share/fonts',
        os.path.expanduser('~/.fonts'),
        os.path.expanduser('~/.local/share/fonts'),
    ])

    font_dirs.append(os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts'))

    font_dirs.extend([
        '/Library/Fonts',
        '/System/Library/Fonts',
        os.path.expanduser('~/Library/Fonts'),
    ])

    fonts_found = set()

    for font_dir in font_dirs:
        if os.path.exists(font_dir):
            for root, dirs, files in os.walk(font_dir):
                for f in files:
                    if f.lower().endswith(('.ttf', '.otf', '.ttc')):
                        font_path = os.path.join(root, f)
                        font_name = os.path.splitext(f)[0]
                        if font_name not in fonts_found:
                            fonts_found.add(font_name)
                            items.append((font_path, font_name, f"Font: {font_name}"))
                if root != font_dir:
                    break

    items.sort(key=lambda x: x[1].lower())
    items = items[:200]

    if not items:
        items.append(('NONE', '(No fonts found)', 'No system fonts found'))

    return items
