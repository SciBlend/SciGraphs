# Reads georeferenced rasters (GeoTIFF): pixel data, geographic metadata and CRS.

import numpy as np
from scigraphs_core.logger import log

_HAS_GDAL = False
try:
    from osgeo import gdal
    _HAS_GDAL = True
except ImportError:
    pass


class GeoRaster:
    """Reader for georeferenced raster files (GeoTIFF).

    Uses GDAL when it is installed. Otherwise falls back to PIL for the pixels
    and hand-parsed TIFF tags for the geographic metadata.
    """
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.dtype = None
        self.width = 0
        self.height = 0
        self.bands = 1
        self.nodata = None
        self.crs = None
        
        # (origin_x, pixel_width, 0, origin_y, 0, pixel_height). pixel_height is
        # usually negative, since y increases downward in image space.
        self.geotransform = None
        
        # In CRS coordinates.
        self.bounds = None
        
        self._load()
    
    def _load(self):
        """Load raster data and metadata."""
        if _HAS_GDAL:
            self._load_gdal()
        else:
            self._load_pure_python()
    
    def _load_gdal(self):
        """Load using GDAL."""
        try:
            ds = gdal.Open(self.filepath)
            if ds is None:
                raise IOError(f"GDAL could not open {self.filepath}")
            
            self.width = ds.RasterXSize
            self.height = ds.RasterYSize
            self.bands = ds.RasterCount
            self.geotransform = ds.GetGeoTransform()
            
            proj = ds.GetProjection()
            if proj:
                from osgeo import osr
                srs = osr.SpatialReference()
                srs.ImportFromWkt(proj)
                self.crs = srs.GetAuthorityCode(None)
                if self.crs:
                    self.crs = f"EPSG:{self.crs}"
            
            # Band 1 carries the elevation.
            band = ds.GetRasterBand(1)
            self.nodata = band.GetNoDataValue()
            self.dtype = gdal.GetDataTypeName(band.DataType).lower()
            self.data = band.ReadAsArray()
            
            self._calculate_bounds()
            
            ds = None  # Close dataset
            
            log(f"GeoRaster loaded (GDAL): {self.width}x{self.height}, {self.dtype}")
            
        except Exception as e:
            log(f"GDAL load failed: {e}, trying pure Python...")
            self._load_pure_python()
    
    def _load_pure_python(self):
        """Load using pure Python (PIL + custom TIFF tag parsing)."""
        try:
            from PIL import Image
            
            img = Image.open(self.filepath)
            self.width, self.height = img.size
            
            mode = img.mode
            if mode == 'I':
                self.data = np.array(img, dtype=np.int32)
                self.dtype = 'int32'
            elif mode == 'I;16' or mode == 'I;16S':
                self.data = np.array(img, dtype=np.int16)
                self.dtype = 'int16'
            elif mode == 'F':
                self.data = np.array(img, dtype=np.float32)
                self.dtype = 'float32'
            elif mode == 'L':
                self.data = np.array(img, dtype=np.uint8)
                self.dtype = 'uint8'
            elif mode in ['RGB', 'RGBA']:
                # Color image: flatten to grayscale and read it as elevation.
                self.data = np.array(img.convert('L'), dtype=np.uint8)
                self.dtype = 'uint8'
                self.bands = 3 if mode == 'RGB' else 4
            else:
                self.data = np.array(img)
                self.dtype = str(self.data.dtype)
            
            self._extract_geotiff_tags(img)
            
            img.close()
            
            log(f"GeoRaster loaded (PIL): {self.width}x{self.height}, {self.dtype}")
            
        except Exception as e:
            log(f"Error loading raster: {e}")
            raise
    
    def _extract_geotiff_tags(self, img):
        """Extract GeoTIFF tags from a PIL image."""
        MODEL_TIEPOINT_TAG = 33922
        MODEL_PIXEL_SCALE_TAG = 33550
        GEO_KEY_DIRECTORY_TAG = 34735
        
        try:
            tiff_tags = img.tag_v2 if hasattr(img, 'tag_v2') else img.tag
            
            # Tie point (i, j, k, x, y, z) maps pixel i, j (usually 0, 0) to the
            # geographic coordinate x, y. Scale is the pixel size in CRS units.
            tiepoint = tiff_tags.get(MODEL_TIEPOINT_TAG)
            scale = tiff_tags.get(MODEL_PIXEL_SCALE_TAG)
            
            if tiepoint and scale:
                if len(tiepoint) >= 6 and len(scale) >= 2:
                    i, j = tiepoint[0], tiepoint[1]
                    x, y = tiepoint[3], tiepoint[4]
                    sx, sy = scale[0], scale[1]
                    
                    # Top-left corner.
                    origin_x = x - (i * sx)
                    origin_y = y + (j * sy)  # Note: y increases upward in geo coords
                    
                    self.geotransform = (origin_x, sx, 0, origin_y, 0, -sy)
                    self._calculate_bounds()
            
            geokeys = tiff_tags.get(GEO_KEY_DIRECTORY_TAG)
            if geokeys and len(geokeys) >= 4:
                # Entries run in groups of four. A projected CRS (3072) wins over
                # a geographic one (2048).
                for i in range(4, len(geokeys), 4):
                    if i + 3 < len(geokeys):
                        key_id = geokeys[i]
                        value = geokeys[i + 3]
                        if key_id == 3072 and value > 0:  # ProjectedCSTypeGeoKey
                            self.crs = f"EPSG:{value}"
                            break
                        elif key_id == 2048 and value > 0:  # GeographicTypeGeoKey
                            self.crs = f"EPSG:{value}"
                        
        except Exception as e:
            log(f"Could not extract GeoTIFF tags: {e}")
    
    def _calculate_bounds(self):
        """Calculate bounding box from geotransform."""
        if self.geotransform is None:
            return
        
        ox, sx, _, oy, _, sy = self.geotransform
        
        # sy is usually negative.
        west = ox
        east = ox + self.width * sx
        north = oy
        south = oy + self.height * sy
        
        if south > north:
            north, south = south, north
        
        self.bounds = {
            'west': west,
            'east': east,
            'south': south,
            'north': north,
        }
    
    def get_pixel_size(self):
        """Get pixel size in CRS units (x, y)."""
        if self.geotransform:
            return abs(self.geotransform[1]), abs(self.geotransform[5])
        return None, None
    
    def get_origin(self):
        """Get origin (top-left corner) in CRS coordinates."""
        if self.geotransform:
            return self.geotransform[0], self.geotransform[3]
        return None, None
    
    def pixel_to_geo(self, col, row):
        """Convert pixel coordinates to geographic coordinates."""
        if self.geotransform is None:
            return None, None
        
        ox, sx, _, oy, _, sy = self.geotransform
        x = ox + col * sx
        y = oy + row * sy
        return x, y
    
    def geo_to_pixel(self, x, y):
        """Convert geographic coordinates to pixel coordinates."""
        if self.geotransform is None:
            return None, None
        
        ox, sx, _, oy, _, sy = self.geotransform
        col = int((x - ox) / sx)
        row = int((y - oy) / sy)
        return col, row
    
    def get_elevation_at(self, x, y):
        """Get elevation value at geographic coordinates."""
        col, row = self.geo_to_pixel(x, y)
        if col is None:
            return None
        
        if 0 <= col < self.width and 0 <= row < self.height:
            value = self.data[row, col]
            if self.nodata is not None and value == self.nodata:
                return None
            return float(value)
        return None
    
    def get_statistics(self):
        """Get statistics of elevation data."""
        valid_data = self.data
        if self.nodata is not None:
            valid_data = self.data[self.data != self.nodata]
        
        if valid_data.size == 0:
            return None
        
        return {
            'min': float(np.nanmin(valid_data)),
            'max': float(np.nanmax(valid_data)),
            'mean': float(np.nanmean(valid_data)),
            'std': float(np.nanstd(valid_data)),
        }
    
    def is_geographic_crs(self):
        """Check if CRS is geographic (lat/lon) or projected."""
        if self.crs is None:
            return True  # Assume geographic if unknown
        
        # WGS84 and the two NAD codes. Web Mercator (3857) and the UTM zones
        # (326xx, 327xx) are projected, so they fall through as False.
        geographic_codes = {'4326', '4269', '4267'}
        
        code = self.crs.replace('EPSG:', '')
        return code in geographic_codes


def load_georaster(filepath):
    """Load a GeoRaster, or return None if it cannot be read."""
    try:
        return GeoRaster(filepath)
    except Exception as e:
        log(f"Failed to load GeoRaster: {e}")
        return None

