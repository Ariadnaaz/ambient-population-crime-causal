import pandas as pd
import numpy as np
from libpysal.weights import Queen
import geopandas as gpd
from shapely import wkt

def generate_W_matrix(city):
    """
    Generate our contiguity matrix W, where each row i corresponds to a spatial unit (hexagon i), each column j corresponds to another unit
    (hexagon j), and the entry W_{ij} gives the influence of unit j on unit i (after row-standardization). So, the row represents
    a focal location, and the columns are its neighbors. Each row sums to 1 after normalization, so it’s a weighted average of neighbors. Then
    the spatial lag (or spillover) of a variable x is defined as y=Wx
    """
    hex_df = pd.read_csv(f'../hexagons_list/{city}_hex_list.csv')
    hex_df["polygon"] = hex_df["polygon"].apply(wkt.loads)
    hex_gdf = gpd.GeoDataFrame(hex_df, geometry="polygon", crs="EPSG:4326")

    # Queen contiguity (shared borders OR vertices)
    W = Queen.from_dataframe(hex_gdf, idVariable="hex_index") 

    # row-standardize so rows sum to 1 (Durbin needs this)
    W.transform = "R"

    # sparse representation (efficient for large grids)
    W_sparse = W.sparse

    # convert to dense numpy array
    W = W_sparse.toarray()

    print("Shape of W:", W.shape)
    np.save(f"{city}_W_matrix.npy", W)
