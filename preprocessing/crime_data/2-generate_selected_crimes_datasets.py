# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""#### Baltimore"""

# load dataset clean
df_clean = pd.read_csv(f'raw_data/Baltimore/Baltimore_crimes_clean_all_2y.csv',low_memory=False,index_col=0)
print("Shape clean dataset: ", df_clean.shape)

# get list of crime categories
group = df_clean['crime_type'].unique().tolist()
print(group)

# maual list of crimes we want to keep based on output from group
list_crimes = ['AGG. ASSAULT','COMMON ASSAULT','AUTO THEFT','BURGLARY','HOMICIDE','ROBBERY','ROBBERY - CARJACKING', 'ROBBERY - COMMERCIAL']

# keep only rows with those 6 crimes
df_filtered = df_clean.copy()
df_filtered = df_filtered[df_filtered['crime_type'].isin(list_crimes)]
print("Shape after selecting the 5 types of crimes: ", df_filtered.shape)

# make naming consistent
df_filtered['crime_type'].replace("AUTO THEFT","Motor Vehicle Theft",inplace=True)
df_filtered['crime_type'].replace("AGG. ASSAULT","Assault",inplace=True)
df_filtered['crime_type'].replace("COMMON ASSAULT","Assault",inplace=True)
df_filtered['crime_type'].replace("ROBBERY - CARJACKING","Robbery",inplace=True)
df_filtered['crime_type'].replace("ROBBERY - COMMERCIAL","Robbery",inplace=True)
df_filtered['crime_type'] = df_filtered['crime_type'].str.title()

# save final dataset
df_filtered.to_csv(f'raw_data/Baltimore/Baltimore_selected_crimes_clean_all.csv')
print("Saved dataset")

"""#### Chicago"""

# load dataset clean
df_clean = pd.read_csv(f'raw_data/Chicago/Chicago_crimes_clean_all_2y.csv',low_memory=False,index_col=0)
print("Shape clean dataset: ", df_clean.shape)

# get list of crime categories
group = df_clean['crime_type'].unique().tolist()
print(group)

# maual list of crimes we want to keep based on output from group
list_crimes = ['ASSAULT','MOTOR VEHICLE THEFT','BURGLARY','HOMICIDE','ROBBERY','BATTERY']

# keep only rows with those 6 crimes
df_filtered = df_clean.copy()
df_filtered['crime_type'].replace("BATTERY","Assault",inplace=True)
df_filtered = df_filtered[df_filtered['crime_type'].isin(list_crimes)]
print("Shape after selecting the 5 types of crimes: ", df_filtered.shape)

# make naming consistent
df_filtered['crime_type'] = df_filtered['crime_type'].str.title()

# save final dataset
df_filtered.to_csv(f'raw_data/Chicago/Chicago_selected_crimes_clean_all.csv')
print("Saved dataset")

"""#### Philadelphia"""

# load dataset clean
df_clean = pd.read_csv(f'raw_data/Philadelphia/Philadelphia_crimes_clean_all_2y.csv',low_memory=False,index_col=0)
print("Shape clean dataset: ", df_clean.shape)

# get list of crime categories
group = df_clean['crime_type'].unique().tolist()
print(group)

# maual list of crimes we want to keep based on output from group
list_crimes = ['Burglary Non-Residential', 'Aggravated Assault No Firearm', 'Aggravated Assault Firearm', 'Burglary Residential', 'Robbery No Firearm', 'Robbery Firearm', 'Other Assaults','Motor Vehicle Theft', 'Homicide - Criminal ', 'Homicide - Criminal', 'Homicide - Justifiable ', 'Homicide - Gross Negligence']

# keep only rows with those 5 crimes
df_filtered = df_clean.copy()
df_filtered = df_filtered[df_filtered['crime_type'].isin(list_crimes)]
print("Shape after selecting the 5 types of crimes: ", df_filtered.shape)

# make naming consistent
assault_list = '|'.join([ 'Aggravated Assault No Firearm', 'Aggravated Assault Firearm','Other Assaults'])
homicide_list = '|'.join(['Homicide - Criminal ', 'Homicide - Criminal', 'Homicide - Justifiable ','Homicide - Justifiable', 'Homicide - Gross Negligence'])
burglary_list = '|'.join(['Burglary Non-Residential','Burglary Residential'])
robbery_list = '|'.join(['Robbery No Firearm','Robbery Firearm'])
df_filtered['crime_type'] = df_filtered['crime_type'].str.replace(assault_list,"Assault",regex=True)
df_filtered['crime_type'] = df_filtered['crime_type'].str.replace(homicide_list,"Homicide",regex=True)
df_filtered['crime_type'] = df_filtered['crime_type'].str.replace(burglary_list,"Burglary",regex=True)
df_filtered['crime_type'] = df_filtered['crime_type'].str.replace(robbery_list,"Robbery",regex=True)
df_filtered['crime_type'] = df_filtered['crime_type'].str.title()

# save final dataset
df_filtered.to_csv(f'raw_data/Philadelphia/Philadelphia_selected_crimes_clean_all.csv')
print("Saved dataset")
