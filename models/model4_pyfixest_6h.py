#pip install git+https://github.com/Ariadnaaz/pyfixest.git

import pandas as pd
import numpy as np
import gc
import pyfixest as pf

"""## 6 hours"""

def load_city_data(city_folder, crime_types): 

    # load footfall data
    df_footfall = pd.read_csv(f"../preprocessing/mobility_data/final_data/{city_folder}_mobility_footfall_hex.csv",index_col=0)

    # load and merge all crime types for the city
    dfs = []
    for crime in crime_types:
        df_crime = pd.read_csv(f"../preprocessing/crime_data/{city_folder}_{crime}_all_final_hex.csv", index_col=0)
        dfs.append(df_crime)

    # merge all crimes
    df_crimes = sum(dfs)  

    # load hex list
    hex_df = pd.read_csv(f'../preprocessing/hexagons_list/{city_folder}_hex_list.csv',index_col=0)
    original_hex_order = hex_df.index.tolist()

    df_crimes = df_crimes.loc[:, df_crimes.columns.isin(hex_df.index)]

    # align columns of crime and footfall data to hex list and fill missing with 0
    df_footfall = df_footfall.reindex(columns=hex_df.index, fill_value=0).fillna(0)
    df_crimes = df_crimes.reindex(columns=hex_df.index, fill_value=0).fillna(0)

    # convert to datetime
    df_footfall = make_datetime_index_6h(df_footfall)
    df_crimes = make_datetime_index_6h(df_crimes)

    return df_footfall, df_crimes, original_hex_order

def make_datetime_index_6h(df):
    """Convert {hour}_{year} index into proper datetime index."""
    df = df.reset_index().rename(columns={"index": "time_key"})
    df[["hour_of_year", "year"]] = df["time_key"].str.split("_", expand=True)
    df["hour_of_year"] = df["hour_of_year"].astype(int)
    df["year"] = df["year"].astype(int)
    df["datetime"] = pd.to_datetime(df["year"].astype(str) + "-01-01") \
                        + pd.to_timedelta(df["hour_of_year"], unit="h")

     # floor to 6-hour bins
    df["datetime"] = df["datetime"].dt.floor("6H")

    return (df.drop(columns=["time_key", "hour_of_year", "year"]).groupby("datetime", as_index=True).sum())

def generate_regression_results_model4(lags, cities, model_name, crime_types, covariates, footfall_factor, placebo_leads):

    for city in cities:
        print(f"\n=== CITY: {city} ===")
        city_results = []

        for crime in crime_types:
            print(f"\nDoing CRIME={crime}...")

            # load spatial weights matrix (row-standardized)
            W = np.load(f"../preprocessing/W_matrix/{city}_W_matrix.npy")
            print("W shape: ", np.shape(W))

            # load data
            df_footfall, df_crime, original_hex_order = load_city_data(city_folder=city, crime_types=crime)
            df_crime = df_crime.astype(np.float32)
            df_footfall = df_footfall.astype(np.float32)

            # keep only hexes with crimes
            hexes_with_crime = df_crime.columns[df_crime.sum(axis=0) > 0]
            df_crime = df_crime[hexes_with_crime]
            df_footfall = df_footfall[hexes_with_crime]

            # verify ordering consistency
            assert list(df_crime.columns) == list(hexes_with_crime), "Column ordering mismatch!"

            # get the indices of the kept hexagons in the ORIGINAL ordering
            keep_indices = [original_hex_order.index(h) for h in hexes_with_crime]
            # subset W matrix to only include rows/columns for kept hexagons
            W = W[np.ix_(keep_indices, keep_indices)]
            assert W.shape[0] == len(hexes_with_crime), "W matrix dimension mismatch!"

            # convert to long format
            crime_long = df_crime.reset_index().melt(id_vars="datetime", var_name="hex_id", value_name="crime")
            footfall_long = df_footfall.reset_index().melt(id_vars="datetime", var_name="hex_id", value_name="footfall")
            df = pd.merge(crime_long, footfall_long, on=["datetime", "hex_id"], how="outer")

            # apply sampling correction
            SAMPLE_CORRECTIONS = {
                "Baltimore": 28.51,
                "Chicago": 36.41,
                "Philadelphia": 39.75,
            }
            df["footfall"] = df["footfall"] * SAMPLE_CORRECTIONS[city]

            del crime_long, footfall_long
            gc.collect()

            # load and expand covariates (daily to 6h blocks)
            def expand_daily_to_6h(df_cov):
                df_cov = df_cov.copy().astype(int)
                df_cov.index = pd.to_datetime(df_cov.index)

                return df_cov.reindex(df_cov.index.repeat(4)).assign( # duplicates each daily row 4 times (once per 6h block)
                        datetime=lambda x: np.repeat(df_cov.index, 4)+ pd.to_timedelta(np.tile([0, 6, 12, 18], len(df_cov)), unit="h")
                    ).set_index("datetime")

            def load_covariates(city):
                df_sports = pd.read_csv(
                    f"../confounders_and_moderators_datasets/{city}_sport_events.csv",
                    index_col=0, parse_dates=True
                )
                df_holidays = pd.read_csv(
                    f"../confounders_and_moderators_datasets/Holidays_{city}.csv",
                    index_col=0, parse_dates=True
                )
                df_extras = pd.read_csv(
                    f"../confounders_and_moderators_datasets/{city}_extra_events.csv",
                    index_col=0, parse_dates=True
                )
                return expand_daily_to_6h(df_sports), expand_daily_to_6h(df_holidays), expand_daily_to_6h(df_extras)

            df_sports_hr, df_holidays_hr, df_extras_hr = load_covariates(city)

            # merge covariates with main df and rename
            df = (df.merge(df_sports_hr, on="datetime", how="left")
                    .merge(df_holidays_hr, on="datetime", how="left")
                    .merge(df_extras_hr, on="datetime", how="left"))
            df.rename(columns={"Event": "extra_event", "Sport_event": "sport_event", "is_holiday": "holiday"}, inplace=True)

            # create a categorical with the correct order to preserve df_footfall column ordering
            hex_order = df_footfall.columns.tolist()
            df["hex_id"] = pd.Categorical(df["hex_id"], categories=hex_order, ordered=True)
            # sort to use the categorical order, not alphabetical
            df = df.sort_values(["datetime", "hex_id"]).reset_index(drop=True)
            # convert back to string for pyfixest compatibility
            df["hex_id"] = df["hex_id"].astype(str)

            # verify the ordering matches what flatten expects
            expected_order = pd.MultiIndex.from_product([df_footfall.index, df_footfall.columns], names=['datetime', 'hex_id'])
            actual_order = pd.MultiIndex.from_arrays([df['datetime'], df['hex_id']])
            assert expected_order.equals(actual_order), "Ordering mismatch for spatial lag assignment!"

            # compute lagged footfall and spatially lagged footfall
            for k in lags:
                df[f'footfall_lag{k}'] = df.groupby("hex_id")["footfall"].shift(k)

                # spatial lag: (W * Footfall_{t-k})_i
                ff_mat = df_footfall.shift(k).to_numpy() # shape: (n_timesteps, n_hexes)
                W_ff = W @ ff_mat.T  # matrix multiplication shape: (n_hexes, n_timesteps)
                result = W_ff.T.flatten(order='C')
                df[f'W_footfall_lag{k}'] = result
                # quick sanity check
                assert len(result) == len(df), "Length mismatch!"
                assert df['datetime'].nunique() * len(hexes_with_crime) == len(result), "Shape mismatch!"

            if placebo_leads == True:
                 # compute leads of footfall and spatial leads footfall
                df[f'footfall_lead1'] = df.groupby("hex_id")["footfall"].shift(-1)

                # spatial lead: (W * Footfall_{t-k})_i
                ff_mat = df_footfall.shift(-1).to_numpy() # shape: (n_timesteps, n_hexes)
                W_ff = W @ ff_mat.T  # matrix multiplication shape: (n_hexes, n_timesteps)
                result = W_ff.T.flatten(order='C')
                df[f'W_footfall_lead1'] = result
                # quick sanity check
                assert len(result) == len(df), "Length mismatch!"
                assert df['datetime'].nunique() * len(hexes_with_crime) == len(result), "Shape mismatch!"


            # spatial lag of lagged crime: (W * Crime_{t-1})_i
            crime_mat = df_crime.shift(1).to_numpy()
            W_crime = W @ crime_mat.T
            df["W_crime_lag1"] = W_crime.T.flatten(order='C')

            df.dropna(inplace=True)

            # outcome and regressors
            df["six_hour_block"] = df["datetime"].dt.hour // 6 # so six_hour_block ∈ {0,1,2,3}
            df["day_of_week"] = df["datetime"].dt.weekday
            df["month"] = df["datetime"].dt.month
            df["year"] = df["datetime"].dt.year

            # build formula string
            if placebo_leads == True:
                regressors = (
                    [f"footfall_lag{k}" for k in lags] +
                    [f"W_footfall_lag{k}" for k in lags] +
                    [f"footfall_lead1"] +
                    [f"W_footfall_lead1"] +
                    ["W_crime_lag1"] +
                    covariates
                    )
            elif placebo_leads == False:
                regressors = (
                    [f"footfall_lag{k}" for k in lags] +
                    [f"W_footfall_lag{k}" for k in lags] +
                    ["W_crime_lag1"] +
                    covariates
                )

            # build the FE part
            fe_parts = ["hex_id", "six_hour_block", "day_of_week", "month", "year"]
            fe_str = " + ".join(fe_parts)

            # combine into fixest-style formula (Y ~ Xs | FEs)
            formula = "crime ~ " + " + ".join(regressors) + " | " + fe_str

            # estimate the model:
            model = pf.fepois(
                formula,
                data = df,
                vcov = {"CRV1": "hex_id"}  # cluster on hex_id
            )

            print(model.summary())

            # ---- extract results exactly like before ----
            crime_str = crime[0] if len(crime) == 1 else 'All'

            # get results as a tidy DataFrame
            results_df = model.tidy()
            for idx, row in results_df.iterrows():
                var = idx
                beta = row['Estimate']
                std_err = row['Std. Error']
                p_val = row['Pr(>|t|)']
                exp_beta_factor = np.exp(beta * footfall_factor)
                city_results.append({
                    "crime_type": crime_str,
                    "variable": var,
                    "beta": beta,
                    "exp_beta_factor": exp_beta_factor,
                    "std_err": std_err,
                    "exp_std_err_factor": exp_beta_factor * footfall_factor * std_err,
                    "p_value": f"{p_val:.8f}",
                    "num_obs": int(model._N),
                    "log-likelihood": model._loglik,
                    "pearson_chi2": model._pearson_chi2,
                    "pseudo_r2": model._pseudo_r2  # McFadden pseudo R2
                })
            gc.collect()

        # save results per city
        city_results_df = pd.DataFrame(city_results)
        city_results_df.to_csv(f"results/Model_with_DL&FE&SE/sample_correction_city_1SD/{city}_Pfixest_results_{model_name}_6H.csv",index=False)
        print(f"Saved {city} results!")

generate_regression_results_model4(lags=[0,1,2],
                                   cities=['Baltimore','Chicago','Philadelphia'], #
                                   model_name='2lags',
                                   crime_types = [
                                        ["Burglary"],
                                        ["Motor Vehicle Theft"],
                                        ["Assault"],
                                        ["Homicide"],
                                        ["Robbery"],
                                        ["Burglary", "Assault", "Robbery", "Homicide", "Motor Vehicle Theft"]
                                    ],
                                   covariates=['extra_event','holiday','sport_event'],
                                   footfall_factor = 36000,
                                   placebo_leads=False)
