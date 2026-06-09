import deweydatapy as ddp

# info about which dataset to download
apikey_ = # your API key
pp_advan_wp = "https://api.deweydata.io/api/v1/external/data/fldr_bpyousrmfggrfubk"

# get dataset metadata
meta = ddp.get_meta(apikey_, pp_advan_wp, print_meta = True)

# get list of files within the given time period
files_df = ddp.get_file_list(apikey_, pp_advan_wp,
                             start_date = '2023-06-01',
                             end_date = '2025-07-01',
                             print_info = True);

# download the files
ddp.download_files(files_df, "raw_data/Advan_data/", skip_exists = True)
