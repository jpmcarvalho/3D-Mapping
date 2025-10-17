import os
import rasterio
import rasterio.mask
from osgeo import osr
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
import numpy as np
import json
import warnings


def calc_volume(input_dem, pts=None, pts_epsg=None, geojson_polygon=None, decimals=4,
                base_method="triangulate", custom_base_z=None):
    try:
        # print(pts_epsg)
        osr.UseExceptions()
        warnings.filterwarnings("ignore", module='scipy.optimize')

        if not os.path.isfile(input_dem):
            raise IOError(f"{input_dem} does not exist")

        crs = None
        with rasterio.open(input_dem) as d:
            if d.crs is None:
                raise ValueError(f"{input_dem} does not have a CRS")
            crs = osr.SpatialReference()
            crs.ImportFromEPSG(d.crs.to_epsg())
        
        if pts is None and pts_epsg is None and geojson_polygon is not None:
            # Read GeoJSON points
            pts = read_polygon(geojson_polygon)
            return calc_volume(input_dem, pts=pts, pts_epsg=4326, decimals=decimals, base_method=base_method, custom_base_z=custom_base_z)
        
        # Convert to DEM crs
        src_crs = osr.SpatialReference()
        src_crs.ImportFromEPSG(pts_epsg)
        transformer = osr.CoordinateTransformation(src_crs, crs)

        dem_pts = [list(transformer.TransformPoint(p[0], p[1]))[:2] for p in pts]

        print("dem_pts")
        print(dem_pts)
        
        # Some checks
        if len(dem_pts) < 2:
            raise ValueError("Insufficient points to form a polygon")

        # Close loop if needed
        if not np.array_equal(dem_pts[0], dem_pts[-1]):
            dem_pts.append(dem_pts[0])
        
        polygon = {"coordinates": [dem_pts], "type": "Polygon"}
        dem_pts = np.array(dem_pts)

        # Remove last point (loop close)
        dem_pts = dem_pts[:-1]
        
        with rasterio.open(input_dem) as d:
            px_w = d.transform[0]
            px_h = d.transform[4]

            # band1 = d.read(1)
            # print('Band1 has shape', band1.shape)
            # height = band1.shape[0]
            # width = band1.shape[1]

            # print("h")
            # print(height)
            # print(width)

            # print("px")
            # print(px_h)
            # print(px_w)

            # cols, rows = np.meshgrid(np.arange(width), np.arange(height))
            # xs, ys = rasterio.transform.xy(d.transform, rows, cols)
            # lons= np.array(xs)
            # lats = np.array(ys)
            # print('lons', lons)
            # print('lats', lats)

            # Area of a pixel in square units
            px_area = abs(px_w * px_h)

            # print("Area:::::")
            # print(px_area)
            # print(polygon)

            rast_dem, transform = rasterio.mask.mask(d, [polygon], crop=True, all_touched=True, indexes=1, nodata=np.nan)
            # print(rast_dem.shape)
            h, w = rast_dem.shape


            # X/Y coordinates in transform coordinates
            ys, xs = np.array(rasterio.transform.rowcol(transform, dem_pts[:,0], dem_pts[:,1]))

            if np.any(xs<0) or np.any(xs>=w) or np.any(ys<0) or np.any(ys>=h):
                raise ValueError("Points are out of bounds")
            
            zs = rast_dem[ys,xs]

            if base_method == "plane":
                # Create a grid for interpolation
                x_grid, y_grid = np.meshgrid(np.linspace(0, w - 1, w), np.linspace(0, h - 1, h))

                # Perform curve fitting
                linear_func = lambda xy, m1, m2, b: m1 * xy[0] + m2 * xy[1] + b
                params, covariance = curve_fit(linear_func, np.vstack((xs, ys)), zs)

                base = linear_func((x_grid, y_grid), *params)
            elif base_method == "triangulate":
                # Create a grid for interpolation
                x_grid, y_grid = np.meshgrid(np.linspace(0, w - 1, w), np.linspace(0, h - 1, h))
                
                # Tessellate the input point set to N-D simplices, and interpolate linearly on each simplex. 
                base = griddata(np.column_stack((xs, ys)), zs, (x_grid, y_grid), method='linear')
            elif base_method == "average":
                base = np.full((h, w), np.mean(zs))
            elif base_method == "custom":
                if custom_base_z is None:
                    raise ValueError("Base method set to custom, but no custom base Z specified")
                base = np.full((h, w), float(custom_base_z))
            elif base_method == "highest":
                base = np.full((h, w), np.max(zs))
            elif base_method == "lowest":
                base = np.full((h, w), np.min(zs))
            else:
                raise ValueError(f"Invalid base method {base_method}")

            base[np.isnan(rast_dem)] = np.nan

            # Calculate volume
            diff = rast_dem - base
            volume = np.nansum(diff) * px_area

            print(volume)

            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots()
            # ax.imshow(base)
            # plt.scatter(xs, ys, c=zs, cmap='viridis', s=50, edgecolors='k')
            # plt.colorbar(label='Z values')
            # plt.title('Debug')
            # plt.show()

            print({'output': np.abs(np.round(volume, decimals=decimals))})

            return np.abs(np.round(volume, decimals=decimals))
    except Exception as e:
        return {'error': str(e)}

def read_polygon(file):
    with open(file, 'r', encoding="utf-8") as f:
        data = json.load(f)

    if data.get('type') == "FeatureCollection":
        features = data.get("features", [{}])
    else:
        features = [data]

    for feature in features:
        if not 'geometry' in feature:
            continue

        # Check if the feature geometry type is Polygon
        if feature['geometry']['type'] == 'Polygon':
            # Extract polygon coordinates
            coordinates = feature['geometry']['coordinates'][0]  # Assuming exterior ring
            return coordinates
    
    raise IOError("No polygons found in %s" % file)



# points = [[-8.872595,40.04939],[-8.872586,40.048942],[-8.870881,40.048915],[-8.870918,40.049433],[-8.872595,40.04939]]

# calc_volume("/home/draco/Downloads/dsm.tif", points, 4326, base_method="triangulate")