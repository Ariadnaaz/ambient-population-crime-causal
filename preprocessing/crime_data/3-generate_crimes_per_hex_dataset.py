import pandas as pd
from shapely import wkt
from shapely.geometry import Point
import numpy as np
import gc
import datetime
from scipy.sparse import lil_matrix
from pandarallel import pandarallel
import os
pandarallel.initialize(nb_workers=min(os.cpu_count(), 12),progress_bar=True)

"""### 1. Generate crime files with hex number"""

def find_polygon(row,df):
    polygon_num = np.nan
    for p_idx,elem in df.iterrows():
      if elem['polygon'].contains(Point(row['longitude'],row['latitude'])):
        polygon_num = elem['hex_index']
        #print(polygon_num)
        break
    return pd.Series({'hex': polygon_num})

def generate_crime_files_with_grid_num(city_name,city_folder,crimes_list):
  print(f"City={city_name}")
  # load crime dataset for that city
  df_crimes = pd.read_csv(f'raw_data/{city_folder}/{city_folder}_selected_crimes_clean_all.csv',index_col=0)

  # load hexagonal polygons
  df_poly = pd.read_csv(f'../confounders_and_moderators_data/{city_folder}_sociodem_hex.csv',index_col=0)
  # convert WKT strings into shapely geometries
  df_poly["polygon"] = df_poly["polygon"].apply(wkt.loads)

  # generate file for each crime
  for crime in crimes_list:
    print("Crime type: ", crime)
    df_crime = df_crimes.copy()
    df_crime = df_crime[df_crime['crime_type'] == crime]
    df_crime.reset_index(drop=True,inplace=True)
    print(f"Shape dataset after selecting only {crime}: ", df_crime.shape)

    print("Finding the hexagon cell for each point...")
    df_final = df_crime.merge(df_crime.parallel_apply(find_polygon,df=df_poly,axis=1),left_index=True, right_index= True)

    # save final dataset
    df_final.to_csv(f'{city_folder}_{crime}_clean_all_grid.csv')
    print("Final dataset saved!\n")

generate_crime_files_with_grid_num(city_name = 'Baltimore',
                                   city_folder = 'Baltimore',
                                   crimes_list = ['Burglary', 'Motor Vehicle Theft','Assault', 'Robbery', 'Homicide']) #

generate_crime_files_with_grid_num(city_name = 'Chicago',
                                   city_folder = 'Chicago',
                                   crimes_list = ['Burglary', 'Motor Vehicle Theft','Assault', 'Robbery', 'Homicide']) #

generate_crime_files_with_grid_num(city_name = 'Philadelphia',
                                   city_folder = 'Philadelphia',
                                   crimes_list = ['Burglary', 'Motor Vehicle Theft','Assault', 'Robbery', 'Homicide']) #

"""### 2. Generate yearly matrix of crime counts per hour"""

# SPARSE VERSION - much faster and doesn't run out of ram

def hour_of_year(dt):
    """Compute the hour of the year from a datetime"""
    beginning_of_year = datetime.datetime(dt.year, 1, 1, tzinfo=dt.tzinfo)
    return int((dt - beginning_of_year).total_seconds() // 3600)

def make_final_grid_files_year_crime(city_folder, crime, years):
    print("City: ", city_folder)

    # load all crimes for the city and crime type
    df_crimes = pd.read_csv(
        f'{city_folder}_{crime}_clean_all_grid.csv',
        index_col=0,
        parse_dates=['crime_date_time'],
        date_parser=pd.to_datetime
    )
    print("Shape whole dataset: ", df_crimes.shape)

    # precompute hour of the year for all crimes
    df_crimes['hour_year'] = df_crimes['crime_date_time'].apply(hour_of_year)

    for y in years:
        print("YEAR: ", y)
        df_year = df_crimes[df_crimes['crime_date_time'].dt.year == y].copy()

        # keep only necessary columns
        df_year = df_year[['hex', 'hour_year']]

        # drop missing hexes and ensure hex IDs are strings
        df_year = df_year[df_year['hex'].notna()]
        df_year['hex'] = df_year['hex'].astype(str)

        print("Shape for this year after cleaning: ", df_year.shape)

        # number of hours in the year
        num_hours_total = int(((datetime.datetime(y+1,1,1) - datetime.datetime(y,1,1)).total_seconds()) // 3600)

        hexes = df_year['hex'].values
        hours = df_year['hour_year'].astype(int).values
        unique_hexes = np.unique(hexes)
        hex_to_idx = {h: i for i, h in enumerate(unique_hexes)}

        # create sparse matrix (rows = hexes, columns = hours)
        mat = lil_matrix((len(unique_hexes), num_hours_total), dtype=int)
        for h, hr in zip(hexes, hours):
            mat[hex_to_idx[h], hr] += 1

        # convert to DataFrame exactly like your old version
        df_final = pd.DataFrame(mat.toarray(), index=unique_hexes, columns=list(range(num_hours_total)))
        df_final.index.name = 'hex'

        # save CSV
        df_final.to_csv(f'{city_folder}_{crime}_{y}_final_grid.csv')
        print("Final dataset saved for year", y)

        # free memory
        del df_year, mat, df_final
        gc.collect()

# make final grid files for each city, year, and crime type
for city in ['Baltimore','Chicago','Philadelphia']:
    for crime in ['Burglary', 'Robbery', 'Homicide','Motor Vehicle Theft', 'Assault']:
        make_final_grid_files_year_crime(city_folder=city,crime=crime,years=[2023,2024,2025])
    print("\n")

"""### 3. Make final grid matrix for each city for each crime all years together"""

def make_final_grid_city_crime(city_folder,crime,years):
  print(f"City: {city_folder} and cime={crime}")

  # concat the grid for each year after renaming columns
  df_all_list = []
  for year in years:
    print(f"Doing year {year}...")
    df_year = pd.read_csv(f'{city_folder}_{crime}_{year}_final_grid.csv',index_col=0)

    # rename columns to indicate year
    df_year.columns = list(map(lambda x: str(x) + f"_{str(year)}", df_year.columns.tolist()))
    print("size: ",df_year.shape)
    df_all_list.append(df_year)

  # concat all years
  print("Concatenating dataframe...")
  df_all = pd.concat(df_all_list,axis=1)
  print("Shape before dropping columns: ", df_all.shape)

  # DROP COLUMNS THAT ARE OUTSIDE OUR STUDY PERIOD
  # define start and end of desired window
  start_window = datetime.datetime(2023, 6, 30)
  end_window   = datetime.datetime(2025, 6, 29, 23)
  # 2023: drop hours before 30 June
  start_2023 = datetime.datetime(2023, 1, 1)
  hours_to_drop_2023 = list(range(int((start_window - start_2023).total_seconds() // 3600)))
  # 2025: drop hours after 29 June
  start_2025 = datetime.datetime(2025, 1, 1)
  end_hour_2025 = int((end_window - start_2025).total_seconds() // 3600)
  hours_to_drop_2025 = list(range(end_hour_2025+1, int((datetime.datetime(2026, 1, 1) - start_2025).total_seconds() // 3600)))
  # merge and drop columns
  cols_to_drop = [f"{h}_2023" for h in hours_to_drop_2023] + [f"{h}_2025" for h in hours_to_drop_2025]
  df_all = df_all.drop(columns=cols_to_drop, errors="ignore")
  print("Shape dataframe after dropping columns: ", df_all.shape)

  # save final output
  df_all.T.to_csv(f"{city_folder}_{crime}_all_final_hex.csv")
  print("Shape final dataframe: ", df_all.T.shape)
  print("File saved!\n")

for city in ['Baltimore','Chicago','Philadelphia']:
    for crime in ['Burglary', 'Motor Vehicle Theft','Assault', 'Robbery', 'Homicide']:
        make_final_grid_city_crime(city_folder=city,
                            crime=crime,
                            years=[2023,2024,2025])
