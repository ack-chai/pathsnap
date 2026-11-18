"""Reproject lon/lat to metric coordinates (UTM 43N for Bengaluru)."""
import numpy as np
from pyproj import Transformer

_T = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)

def to_metric(lon, lat):
    x, y = _T.transform(lon, lat)
    return np.column_stack([x, y])
