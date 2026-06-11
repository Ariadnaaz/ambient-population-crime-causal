import pandas as pd

"""## Generate final hex list by keeping hex common both in soc and mob datasets"""

for city_folder in ['Baltimore','Chicago','Philadelphia']:

    # load mobility hex
    mob_df = pd.read_csv(f'../mobility_data/final_data/{city_folder}_mobility_footfall_hex.csv',index_col=0)
    mob_hex = pd.DataFrame(mob_df.columns.values).rename(columns={0:'hex_index'})
    print("shape mob: ", mob_hex.shape)
    # load sociodemographic hex
    soc_df = pd.read_csv(f'../../confounders_and_moderators_datasets/{city_folder}_sociodem_hex.csv', index_col=0)
    print("shape soc: ", soc_df.shape)

    # check if they are equal already
    print(set(mob_hex["hex_index"]) == set(soc_df["hex_index"]))

    # calculate difference between the two dataframes
    diff1 = mob_hex[~mob_hex["hex_index"].isin(soc_df["hex_index"])]
    diff2 = soc_df[~soc_df["hex_index"].isin(mob_hex["hex_index"])]
    print("hex in mob but not in soc: ",diff1.shape)
    print("hex in soc but not in mob: ",diff2.shape)

    # get final list keeping only common hex
    final_hex = soc_df[soc_df["hex_index"].isin(mob_hex["hex_index"])]
    final_hex = final_hex[['hex_index','polygon']].set_index('hex_index')

    # save final list as csv
    final_hex.to_csv(f'{city_folder}_hex_list.csv')
    print(f"Final shape {city_folder}: {final_hex.shape}")
