import numpy as np
import pandas as pd
import geopandas as gpd
import pickle
import osmnx as ox
import h3
import os
import datetime
from shapely.geometry import Polygon, Point
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=min(os.cpu_count(), 12),progress_bar=True)

"""## Useful functions"""

def extract_multipolygon_city(file_path,city_name):
    '''
    Extracts the entry in the geojson file corresponding to the city selected and outputs the
    corresponding geodataframe with the multipolygon.

        Parameters:
            file_path (str): File path to the geojson
            city_name (str): Name of the city we selected

        Returns:
            feature (geopandas): The geopandas dataframe for that city
    '''
    d = pd.read_json(file_path)
    for feature in d["features"]:
        if feature[0]['properties']['city'] == city_name:
            return gpd.GeoDataFrame.from_features(feature)

def generate_hex_grid(min_lat, min_lon, max_lat, max_lon, resolution, buffer_deg=0.01): # add a small buffer (a bit of latitude/longitude) to each boundary coordinate to ensure that edge hexagons are included
    polygon = {
        "type": "Polygon",
        "coordinates": [[
            [min_lon - buffer_deg, min_lat - buffer_deg],
            [min_lon - buffer_deg, max_lat + buffer_deg],
            [max_lon + buffer_deg, max_lat + buffer_deg],
            [max_lon + buffer_deg, min_lat - buffer_deg],
            [min_lon - buffer_deg, min_lat - buffer_deg]
        ]]
    }
    return list(h3.polyfill(polygon, resolution, geo_json_conformant=True))

def h3_to_polygon(h):
    boundary = h3.h3_to_geo_boundary(h, geo_json=True)
    return Polygon(boundary)

# keep hexagons that intersect at least 20% with the city polygon
def intersects_threshold(poly, city_geom, threshold):
    if not poly.intersects(city_geom):
        return False
    inter_area = poly.intersection(city_geom).area
    poly_area = poly.area
    ratio = inter_area / poly_area
    return ratio >= threshold

# function that removes rows from a DataFrame where the polygon is more than `threshold` percent covered by water.
def drop_water_dominated_cells(df, selected_city, threshold):
    # set the state of each city
    city_states = {
        'Baltimore': 'Maryland',
        'Chicago': 'Illinois',
        'Philadelphia': 'Pennsylvania'
    }

    # get the city's polygon
    city_gdf = ox.geocode_to_gdf(f"{selected_city}, {city_states[selected_city]}")
    city_polygon = city_gdf.geometry.iloc[0]

    # get water features
    water_features = ox.features_from_polygon(city_polygon, tags={'natural': 'water', 'place': 'sea'})
    all_water_union = shapely.unary_union(water_features['geometry'])

    # identify rows to drop
    rows_to_drop = []
    for i, row in df.iterrows():
        poly = row['polygon']
        inter_area = poly.intersection(all_water_union).area
        poly_area = poly.area
        if poly_area > 0 and (inter_area / poly_area) > threshold:
            rows_to_drop.append(i)

    # drop and return
    return df.drop(index=rows_to_drop).reset_index(drop=True)

# generate hexagonal grid per city, using the functions above
def generate_hexagons(selected_city, resolution):
    # extract city multiplygon
    gdf = extract_multipolygon_city(
        file_path='../../city_multipolygons.geojson',
        city_name=selected_city
    )

    # get bounding box
    min_lon, min_lat, max_lon, max_lat = gdf.total_bounds

    print(f"Generating H3 hexagons for bounding box...")
    hex_list = generate_hex_grid(min_lat, min_lon, max_lat, max_lon, resolution)

    # convert hex IDs to polygons and store them in a DataFrame
    polygon_list = [h3_to_polygon(hex) for hex in hex_list]
    hex_df = pd.DataFrame(polygon_list,columns=['polygon'])
    print("Shape of hexagon cell dataframe:", hex_df.shape)

    # give index name to use later
    hex_df['hex_index'] = hex_list

    # keep only hexes that intersect the city boundary 100%
    hex_df = hex_df[hex_df['polygon'].apply(lambda poly: intersects_threshold(poly, gdf.unary_union, threshold=0.999))]
    print("Shape of hexagon cell dataframe after filtering city borders:", hex_df.shape)

    # remove hexagons that are more than 80% in water
    hex_df = drop_water_dominated_cells(hex_df, selected_city, 0.8)
    print("Shape of hexagon cell dataframe after filtering water:", hex_df.shape)

    hex_gdf = gpd.GeoDataFrame(hex_df, geometry='polygon', crs="EPSG:4326")

    # remove manually specified hexagons by index
    if selected_city == 'Baltimore':
        indexes_to_remove = ['882aa8c441fffff', '882aa8c409fffff', '882aa8c451fffff']
    elif selected_city == 'Chicago':
        indexes_to_remove = ['8826645755fffff','882664cb67fffff','882664cb65fffff','882664cb6dfffff','88275936b3fffff',
                             '88275934d9fffff','882664cb61fffff', '882664cb69fffff', '882664cb6bfffff', '882664cb63fffff',
                             '882664cb0dfffff','882664cb47fffff', '882664cb45fffff', '882664cb41fffff', '882664cb43fffff',
                             '88275936bbfffff', '8875936b7fffff', '8826645757fffff']
    elif selected_city == 'Philadelphia':
        indexes_to_remove = []

    print(f"Removing hexes with indexes: {indexes_to_remove}")
    hex_gdf = hex_gdf[~hex_gdf['hex_index'].isin(indexes_to_remove)]
    print("Shape hexagons df after removing some hexagons manually:", hex_gdf.shape)

    return hex_gdf

def find_polygon_mobility(row,df):
    for p_idx,elem in df.iterrows():
      if elem['polygon'].contains(Point(row['longitude'],row['latitude'])):
        polygon_num = elem['hex_index']
        break
      else:
        polygon_num = np.nan
    return pd.Series({'hex': polygon_num})

def make_list_unique_pois(city_selected):
  df1 = pd.read_csv(f'raw_data/Mobility_data_2023_week_26.csv',index_col=0)
  df1.set_index('placekey',inplace=True,drop=True)
  # select only rows for that city
  df1 = df1[df1['city'] == city_selected]
  # keep only location coordinate (we don't care about other variables)
  df1 = df1[['latitude','longitude']]
  for year in ['2023','2024','2025']:
    print(f"Doing year {year}...")
    max_week = 26 if year == '2025' else 52
    min_week = 27 if year == '2023' else 1 # 27 because we already did 26 to start the dataframe
    for week in range(min_week,max_week+1):
      df2 = pd.read_csv(f'raw_data/Mobility_data_{year}_week_{week:02}.csv',index_col=0)
      df2.set_index('placekey',inplace=True,drop=True)
      df2 = df2[df2['city'] == city_selected]
      df2 = df2[['latitude','longitude']]
      df1 = pd.concat([df1, df2])
      df1 = df1[~df1.index.duplicated(keep='first')]
    print(f"Shape df1 after {year}: ", df1.shape)
  return df1

"""##1. Generate mapping between pois and grid num"""

def generate_mobility_pois_to_grid_num_map(selected_city): # needs the weekly files of all years
  # create full dataset of unique POIs for this city
  print("Generate dataframe of unique pois... ")
  df_pois = make_list_unique_pois(city_selected=selected_city)
  df_pois.reset_index(inplace=True)
  print("Size pois dataset: ", df_pois.shape)

  # remove POIs that aren't within city's borders
  gdf = extract_multipolygon_city(
    file_path='../../city_multipolygons.geojson',
    city_name=selected_city
    )
  city_border = shapely.unary_union(gdf.geometry)
  df_pois = df_pois[df_pois.apply(lambda row: city_border.contains(Point(row['longitude'], row['latitude'])), axis=1)]
  print("Size pois dataset after filtering city border: ", df_pois.shape)

  # generate hexagons
  print("Generate hexagons...")
  gdf_poly = generate_hexagons(selected_city=selected_city, resolution=8)
  print("Shape hexagons polygons dataframe: ", gdf_poly.shape)

  print("Finding the hexagon for each POI...")
  test = df_pois.parallel_apply(find_polygon_mobility,df=gdf_poly,axis=1)
  df_final = df_pois.merge(test,left_index=True, right_index= True)
  print("Shape dataframe after adding cell number: ", df_final.shape)

  print("Generating mapping dictionary")
  mapping = dict(df_final[['placekey', 'hex']].values)


  city_folder = selected_city
  with open(f'maps/{city_folder}_mobility_poi_hex_map.pkl', 'wb') as fp:
    pickle.dump(mapping, fp)
    print('POI to grid cell dictionary saved successfully to file!')

for city in ['Baltimore','Chicago','Philadelphia']:
    print("CITY: ", city)
    generate_mobility_pois_to_grid_num_map(city)
    print("\n")

"""## Get mobility data per hexagon using the map"""

def hour_of_week(year, week):
    # Get the Monday of the given ISO week
    start_date = datetime.datetime.fromisocalendar(year, week, 1)
    end_date = start_date + datetime.timedelta(days=7)

    beginning_of_year = datetime.datetime(year, 1, 1)
    start = int((start_date - beginning_of_year).total_seconds() // 3600)
    end = int((end_date - beginning_of_year).total_seconds() // 3600)

    return (start, end)

def make_grid_files_year_mobility(city_name,city_folder,years): # make sure it's correct when using again (had to rewrite bc it didn't save)
  print(f"CITY={city_name}")
  # we assume output folder is inside input folder
  # Read dictionary pkl file
  with open(f'maps/{city_folder}_mobility_poi_hex_map.pkl', 'rb') as fp:
      map_dic = pickle.load(fp)
  #map_dic

  for year in years:
    print(f"Doing year {year}...")
    # load rows for that week for the first week
    min_week = 26 if year == 2023 else 1
    df1 = pd.read_csv(f'raw_data/Mobility_data_{year}_week_{min_week:02}.csv',index_col=0) #parse_dates=['date_range_start','date_range_end']
    df1 = df1[df1['city']==city_name]

    # add hex number and only keep rows with hex num
    df1['hex'] = df1['placekey'].map(map_dic)
    df1 = df1.dropna(subset=['hex'])  # drop rows with NaN in 'hex'
    print("Shape after dropping rows without hexagon num:",df1.shape)

    # group by hex and sum values in the visits per hour to get total number of people in each hex per hour
    df_grouped = df1.groupby(['hex'])[np.arange(1,169).astype('str').tolist()].sum()
    df_grouped = df_grouped.astype(int)

    # rename columns
    start, end = hour_of_week(year,min_week) # start of 2023 has extra days bc week 26 starts at day 26 and not 30th of June
    df_grouped.columns = list(range(start,end))
    df1 = df_grouped.copy()
    max_week = 26 if str(year) == '2025' else 52
    for week in range(min_week+1,max_week+1): # min_week+1 because first week is loaded earlier
      print("week:", week)
      # load rows for that week for the first week
      df2 = pd.read_csv(f'raw_data/Mobility_data_{year}_week_{week:02}.csv',index_col=0)
      df2 = df2[df2['city']==city_name]

      # add cell number and only keep rows with cell num
      df2['hex'] = df2['placekey'].map(map_dic)
      df2 = df2.dropna(subset=['hex'])  # drop rows with NaN in 'hex'

      # group by cell and sum values in the visits per hour
      df_grouped = df2.groupby(['hex'])[np.arange(1,169).astype('str').tolist()].sum()
      df_grouped = df_grouped.astype(int)

      # rename columns
      start, end = hour_of_week(year,week)
      df_grouped.columns = list(range(start,end))
      df2 = df_grouped.copy()
      df1 = pd.concat([df1,df2],axis=1)

    df_year = df1.copy()
    df_year.to_csv(f"final_data/{city_folder}_mobility_{year}_footfall_hex.csv") # make sure output folder exists
    print("Shape final dataframe: ", df_year.shape)
    print("Saved file!")

# get the mobility dataset at hexagon level per city per year
make_grid_files_year_mobility(city_name='Baltimore',city_folder='Baltimore',years=[2023,2024,2025])
make_grid_files_year_mobility(city_name='Chicago',city_folder='Chicago',years=[2023,2024,2025])
make_grid_files_year_mobility(city_name='Philadelphia',city_folder='Philadelphia',years=[2023,2024,2025])


def make_final_grid_city_mobility(city_folder):
  print("City:", city_folder)

  # concat the grid for each year after renaming columns
  df_all_list = []
  for year in [2023,2024,2025]:
    print(f"Doing year {year}...")
    df_year = pd.read_csv(f"final_data/{city_folder}_mobility_{year}_footfall_hex.csv",index_col=0)
    # rename the columns to indicate year
    df_year.columns = [f"{c}_{year}" for c in df_year.columns]
    df_all_list.append(df_year)

  print("Concatenating and saving final dataframe...")
  df_all = pd.concat(df_all_list,axis=1)
  #rename first 48h of 2025 that are actually last 48 hours of 2024
  df_all.rename(columns={f"{i}_2024": f"{8736 + (i + 48)}_2025" for i in range(-48, 0)},inplace=True)
  # drop first days of 2023 so that it starts the 30th of June and not the 26th
  df_all = df_all.drop(columns=[f"{i}_2023" for i in range(4224, 4320)])
  df_all.T.to_csv(f"final_data/{city_folder}_mobility_footfall_hex.csv")
  print("Shape final dataframe: ", df_all.T.shape)

# merge the yearly files into one file per city
for city in ['Baltimore', 'Chicago', 'Philadelphia']:
    make_final_grid_city_mobility(city)
