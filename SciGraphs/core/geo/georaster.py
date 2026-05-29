# GeoRaster module for reading georeferenced raster files
#
# Handles GeoTIFF reading with GDAL fallback to pure Python (Tyf/PIL)
# Extracts geographic metadata, pixel data, and coordinate reference system

import numpy as np
from ...utils.logger import log

# GDAL availability flag
_HAS_GDAL = False
try:
    from osgeo import gdal
    _HAS_GDAL = True
except ImportError:
    pass


class GeoRaster:
    """
    Reader for georeferenced raster files (GeoTIFF).
    
    Supports GDAL for optimal performance, with fallback to pure Python
    using PIL/Pillow for pixel data and custom TIFF tag parsing for
    geographic metadata.
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
        
        # Geotransform: (origin_x, pixel_width, 0, origin_y, 0, pixel_height)
        # pixel_height is typically negative (y increases downward in image)
        self.geotransform = None
        
        # Bounding box in CRS coordinates
        self.bounds = None
        
        self._load()
    
    def _load(self):
        """Load raster data and metadata."""
        if _HAS_GDAL:
            self._load_gdal()
        else:
            self._load_pure_python()
    
    def _load_gdal(self):
        """Load using GDAL (preferred method)."""
        try:
            ds = gdal.Open(self.filepath)
            if ds is None:
                raise IOError(f"GDAL could not open {self.filepath}")
            
            self.width = ds.RasterXSize
            self.height = ds.RasterYSize
            self.bands = ds.RasterCount
            self.geotransform = ds.GetGeoTransform()
            
            # Get CRS
            proj = ds.GetProjection()
            if proj:
                from osgeo import osr
                srs = osr.SpatialReference()
                srs.ImportFromWkt(proj)
                self.crs = srs.GetAuthorityCode(None)
                if self.crs:
                    self.crs = f"EPSG:{self.crs}"
            
            # Read first band (elevation data)
            band = ds.GetRasterBand(1)
            self.nodata = band.GetNoDataValue()
            self.dtype = gdal.GetDataTypeName(band.DataType).lower()
            self.data = band.ReadAsArray()
            
            # Calculate bounds
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
            
            # Get bit depth and convert to numpy array
            mode = img.mode
            if mode == 'I':
                # 32-bit signed integer
                self.data = np.array(img, dtype=np.int32)
                self.dtype = 'int32'
            elif mode == 'I;16' or mode == 'I;16S':
                # 16-bit integer
                self.data = np.array(img, dtype=np.int16)
                self.dtype = 'int16'
            elif mode == 'F':
                # 32-bit float
                self.data = np.array(img, dtype=np.float32)
                self.dtype = 'float32'
            elif mode == 'L':
                # 8-bit grayscale
                self.data = np.array(img, dtype=np.uint8)
                self.dtype = 'uint8'
            elif mode in ['RGB', 'RGBA']:
                # Color image, convert to grayscale for elevation
                self.data = np.array(img.convert('L'), dtype=np.uint8)
                self.dtype = 'uint8'
                self.bands = 3 if mode == 'RGB' else 4
            else:
                # Fallback
                self.data = np.array(img)
                self.dtype = str(self.data.dtype)
            
            # Try to extract geotransform from TIFF tags
            self._extract_geotiff_tags(img)
            
            img.close()
            
            log(f"GeoRaster loaded (PIL): {self.width}x{self.height}, {self.dtype}")
            
        except Exception as e:
            log(f"Error loading raster: {e}")
            raise
    
    def _extract_geotiff_tags(self, img):
        """Extract GeoTIFF tags from PIL image."""
        # GeoTIFF tag IDs
        MODEL_TIEPOINT_TAG = 33922
        MODEL_PIXEL_SCALE_TAG = 33550
        GEO_KEY_DIRECTORY_TAG = 34735
        
        try:
            tiff_tags = img.tag_v2 if hasattr(img, 'tag_v2') else img.tag
            
            # Model tie point: maps pixel to geographic coordinate
            tiepoint = tiff_tags.get(MODEL_TIEPOINT_TAG)
            
            # Pixel scale: size of each pixel in CRS units
            scale = tiff_tags.get(MODEL_PIXEL_SCALE_TAG)
            
            if tiepoint and scale:
                # tiepoint format: (i, j, k, x, y, z)
                # i, j = pixel coordinates (usually 0, 0)
                # x, y = geographic coordinates of that pixel
                if len(tiepoint) >= 6 and len(scale) >= 2:
                    i, j = tiepoint[0], tiepoint[1]
                    x, y = tiepoint[3], tiepoint[4]
                    sx, sy = scale[0], scale[1]
                    
                    # Calculate origin (top-left corner)
                    origin_x = x - (i * sx)
                    origin_y = y + (j * sy)  # Note: y increases upward in geo coords
                    
                    self.geotransform = (origin_x, sx, 0, origin_y, 0, -sy)
                    self._calculate_bounds()
            
            # Try to get CRS from GeoKey directory
            geokeys = tiff_tags.get(GEO_KEY_DIRECTORY_TAG)
            if geokeys and len(geokeys) >= 4:
                # Parse GeoKey directory to find projected/geographic CRS
                # Key 2048 = GeographicTypeGeoKey, 3072 = ProjectedCSTypeGeoKey
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
        
        # sy is typically negative
        west = ox
        east = ox + self.width * sx
        north = oy
        south = oy + self.height * sy
        
        # Ensure north > south
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
        
        # EPSG:4326 is WGS84 (geographic)
        # EPSG:3857 is Web Mercator (projected)
        # EPSG:326xx and 327xx are UTM zones (projected)
        geographic_codes = {'4326', '4269', '4267'}
        
        code = self.crs.replace('EPSG:', '')
        return code in geographic_codes


def load_georaster(filepath):
    """
    Convenience function to load a GeoRaster.
    
    Args:
        filepath: Path to GeoTIFF file
    
    Returns:
        GeoRaster instance or None on error
    """
    try:
        return GeoRaster(filepath)
    except Exception as e:
        log(f"Failed to load GeoRaster: {e}")
        return None

