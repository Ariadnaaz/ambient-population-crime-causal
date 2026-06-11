import pandas as pd
import numpy as np
import os
import datetime
import re
import gzip
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# set folder path where the raw mobility data is stored
folder_path = 'raw_data/Advan_data'

# function to generate the first Mondays of each week in the given date range
def generate_mondays(start_date, end_date):
    mondays = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() == 0:  # Monday
            mondays.append(current_date)
        current_date += datetime.timedelta(days=7)  # increment by 7 days to get the next Monday
    return mondays

# generate the expected dates
start_date = datetime.datetime(2023, 6, 26)
end_date = datetime.datetime(2025, 6, 29)
mondays = generate_mondays(start_date, end_date)
dates_str = set(date.strftime('%Y-%m-%d') for date in mondays)
print("dates str: ",dates_str)

# initialize a dictionary to store file names
file_names = {date: [] for date in dates_str}

# function to count files per week
def file_names_per_week(directory, file_names):
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            # Extract the date from the filename using regex
            match = re.search(r'(\d{4}-\d{2}-\d{2})--patterns_weekly_\d+\.csv', filename)
            if match:
                date_str = match.group(1)
                if date_str in file_names:  # Quick lookup using set
                    file_names[date_str].append(filename)
    return file_names

# get file names
dic_filenames = file_names_per_week(folder_path, file_names)
print(dic_filenames)

# helper DF used to select only these cities
list_cities = [['Baltimore', 'MD'], ['Chicago', 'IL'], ['Philadelphia', 'PA']]
drp = pd.DataFrame(list_cities, columns=['city', 'region']).assign(key=1)

columns_to_keep = ['placekey', 'parent_placekey', 'top_category',
       'latitude', 'longitude', 'city', 'region','visits_by_each_hour',
       'open_hours', 'category_tags', 'opened_on', 'closed_on','location_name','naics_code','median_dwell']

# function to process each file
def process_file(file_name):
    try:
        df = pd.read_csv(folder_path + "/" + file_name, compression="gzip",usecols=columns_to_keep)
        df_filtered = df.merge(drp, how='inner', on=['city', 'region'])
        #df_filtered.drop(columns=columns_to_drop, inplace=True)
        return df_filtered
    except (gzip.BadGzipFile, pd.errors.EmptyDataError) as e:
        print(f"Error processing file {file_name}: {e}")
        return None

# function to process files for a week
def process_week(week,year):
    date_week = str(pd.to_datetime(f"{year}-W{week}-1", format='%G-W%V-%u'))[:10]
    print(f"Number of files for week {week}, so {date_week}:", len(dic_filenames[date_week]))

    df_all = []

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, file_name): file_name for file_name in dic_filenames[date_week]}

        # using tqdm to display a progress bar for file processing
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing week {week}"):
            result = future.result()
            if result is not None:
                df_all.append(result)

    if df_all:
        df_week = pd.concat(df_all, ignore_index=True)
        print(np.shape(df_week))

        df_week.dropna(subset=['visits_by_each_hour'],inplace=True)

        # clean up visits_by_each_hour and turn into a list of integers
        df_week['visits_by_each_hour'] = [str(x).replace('[','').replace(']','').replace('"','').replace('\\', '').split(",") for x in df_week['visits_by_each_hour']]
        df_week['visits_by_each_hour'] = df_week['visits_by_each_hour'].apply(lambda lst: list(map(int, lst)))

        # make column with number of hours in visits_by_each_hour to check if there are 168
        df_week['Number_hours'] = df_week['visits_by_each_hour'].apply(len)

        # keep only rows with 168 values in the visits_by_each_hour
        df_filtered = df_week[df_week['Number_hours'] == 168]
        df_filtered.reset_index(drop=True,inplace=True)
        print("Size dataset after keeping only entries with 168 hours: ", df_filtered.shape)

        # generate columns for each of the 168 hours of the week
        df1 = pd.DataFrame(df_filtered['visits_by_each_hour'].tolist(),columns=list(range(1,169)))

        # merge the two dataframes
        df_complete = pd.concat([df_filtered, df1], axis=1)
        df_complete.to_csv(f"raw_data/Mobility_data_{year}_week_{week:02}.csv")
        print(f"Original size for week {week}: ", df_complete.shape)

# get weekly files for the second half of 2023
year = 2023
for i in range(26, 53):
    process_week(week=i, year=year)
    print("\n")

# get weekly files for 2024
year = 2024
for i in range(1, 53):
    process_week(week=i, year=year)
    print("\n")

# get weekly files for the first half of 2025
year = 2025
for i in range(1, 27):
    process_week(week=i, year=year)
    print("\n")
