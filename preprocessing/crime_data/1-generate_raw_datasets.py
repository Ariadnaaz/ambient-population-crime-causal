import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from statistics import mean

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

"""## First clean raw datasets to select data from 30th June 2023 to 29th June 2025

### Baltimore
"""
# load raw dataset
df = pd.read_csv('raw_data/Baltimore/NIBRS_GroupA_Crime_Data_8914151926081121180.csv',low_memory=False)
print("Size dataframe initially: ", df.shape)

# change type of CrimeDateTime from object to datetime
df['CrimeDateTime'] = df['CrimeDateTime'].map(lambda x: x.replace("+00", ""))
df['CrimeDateTime'] = pd.to_datetime(df['CrimeDateTime'],format="%m/%d/%Y %I:%M:%S %p",errors = 'coerce') # we mark as NaT dates that are too old given datetime lowerbound

# sort by date in ascending order
df.sort_values(by='CrimeDateTime', inplace = True)

# drop rows with with NaT in CrimeDateTime column
df.dropna(subset='CrimeDateTime',inplace=True)
print("Shape dataframe after removing dates too old: ", df.shape)

# make compatible
df.rename(columns={'CrimeDateTime': 'crime_date_time', 'Description':'crime_type','Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True)

# keep only relevant columns
df = df[['crime_date_time','crime_type','latitude','longitude']]

# reset index
df.reset_index(drop=True,inplace=True)

# select dates
lower_bound = "2023/06/30 00:00:00"
upper_bound = "2025/06/30 00:00:00" # last full day is the 29th
df_filter = df.copy()
df_filter = df_filter.loc[(df_filter["crime_date_time"] >= lower_bound) & (df_filter["crime_date_time"] < upper_bound)]
df_filter.reset_index(drop=True,inplace=True)
print("Shape dataframe after selecting crimes from 30th June 2023 to 29th June 2025 incl.", df_filter.shape)

# remove points that are outside the city borders
# extract multipolygon of the city
gdf = extract_multipolygon_city(file_path='../../city_multipolygons.geojson',city_name='Baltimore')

# remove points that aren't within the multipolygon
df_clean = df_filter.copy()
for i, entry in df_filter.iterrows():
    if gdf['geometry'].contains(Point(entry['longitude'], entry['latitude']))[0]:
        None
    else:
        df_clean = df_clean.drop(df_filter.index[i])

# reset index
df_clean.reset_index(drop=True,inplace=True)

df_clean.to_csv("raw_data/Baltimore/Baltimore_crimes_clean_all_2y.csv")
print("Shape final dataframe: ", df_clean.shape)

"""### Chicago"""
# load raw dataset
df = pd.read_csv('raw_data/Chicago/Crimes_-_2001_to_Present_20251024.csv',low_memory=False)
print("Size dataframe initially: ", df.shape)

# change type of Date column from object to datetime
df['Date'] = pd.to_datetime(df['Date'],format='%m/%d/%Y %I:%M:%S %p') # we mark as NaT dates that are too old given datetime lowerbound

# make compatible
df.rename(columns={'Date': 'crime_date_time', 'Primary Type':'crime_type','Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True)

# keep only relevant columns
df = df[['crime_date_time','crime_type','latitude','longitude']]

# select dates
lower_bound = "2023/06/30 00:00:00"
upper_bound = "2025/06/30 00:00:00" # last full day is the 29th
df_years = df.copy()
df_years = df_years.loc[(df_years["crime_date_time"] >= lower_bound) & (df_years["crime_date_time"] < upper_bound)]
print("Shape dataframe after selecting crimes from 30th June 2023 to 29th June 2025 incl.", df_years.shape)

# reset index
df_years.reset_index(drop=True,inplace=True)

# extract multipolygon of the city
gdf = extract_multipolygon_city(file_path='../../city_multipolygons.geojson',city_name='Chicago')

# remove points that aren't within the multipolygon
df_clean = df_years.copy()
for i, entry in df_years.iterrows():
    if gdf['geometry'].contains(Point(entry['longitude'], entry['latitude']))[0]:
        None
    else:
        df_clean = df_clean.drop(df_years.index[i])

# reset index
df_clean.reset_index(drop=True,inplace=True)

# save final dataset
df_clean.to_csv("raw_data/Chicago/Chicago_crimes_clean_all_2y.csv")
print("Shape final dataframe: ", df_clean.shape) # takes around 57 min to run

"""### Philadelphia"""
# load raw datasets
df1 = pd.read_csv('raw_data/Philadelphia/incidents_part1_part2_2023.csv',low_memory=False)
df2 = pd.read_csv('raw_data/Philadelphia/incidents_part1_part2_2024.csv',low_memory=False)
df3 = pd.read_csv('raw_data/Philadelphia/incidents_part1_part2_2025.csv',low_memory=False)
df = pd.concat([df1,df2,df3])
print("Size dataframe initially: ", df.shape)

# change type of CrimeDateTime from object to datetime
df['dispatch_date_time'] = df['dispatch_date_time'].map(lambda x: x.replace("+00", ""))
df['dispatch_date_time'] = pd.to_datetime(df['dispatch_date_time'],format='%Y-%m-%d %H:%M:%S',errors = 'coerce')

# drop rows with with NaT in CrimeDateTime column
df.dropna(subset='dispatch_date_time',inplace=True)
print("Shape dataframe after removing dates too old: ", df.shape)

# reset in order to have the indexes starting from 0
df.reset_index(drop=True,inplace=True)

# make compatible
df.rename(columns={'dispatch_date_time': 'crime_date_time', 'text_general_code':'crime_type','lat':'latitude','lng':'longitude'}, inplace=True)

# keep only relevant columns
df = df[['crime_date_time','crime_type','latitude','longitude']]

# remove points that are outside the city borders
# extract multipolygon of the city
gdf = extract_multipolygon_city(file_path='../../city_multipolygons.geojson',city_name='Philadelphia')

# remove points that aren't within the multipolygon
#df.reset_index(drop=True,inplace=True)
df.reset_index(drop=True,inplace=True)
df_clean = df.copy()
for i, entry in df.iterrows():
    if gdf['geometry'].contains(Point(entry['longitude'], entry['latitude']))[0]:
        None
    else:
        df_clean = df_clean.drop(df.index[i])

# reset index
df_clean.reset_index(drop=True,inplace=True)

df_clean.to_csv(f"raw_data/Philadelphia/Philadelphia_crimes_clean_all_2y.csv")
print("Shape final dataframe: ", df_clean.shape)
