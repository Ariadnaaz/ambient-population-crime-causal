import pandas as pd
import numpy as np
import pyfixest as pf
from joblib import Parallel, delayed
import sys
import gc
from scipy.stats import norm

# parallelized cluster bootstrap function
def single_bootstrap_iteration(
    b, grouped, clusters,
    formula1_lag0, formula1_lag1,
    formula1_Wlag0, formula1_Wlag1,
    formula_Wcrime,
    formula2,
    seed
):
    # unique seed per iteration
    np.random.seed(seed + b)

    # sample clusters with replacement
    sampled_clusters = np.random.choice(clusters, size=len(clusters), replace=True)
    boot_df = pd.concat([df[df["hex_id"] == c] for c in sampled_clusters], axis=0, ignore_index=True)

    try:
        model1_lag0 = pf.feols(formula1_lag0, data=boot_df)
        model1_lag1 = pf.feols(formula1_lag1, data=boot_df)
        model1_Wlag0 = pf.feols(formula1_Wlag0, data=boot_df)
        model1_Wlag1 = pf.feols(formula1_Wlag1, data=boot_df)
        model1_Wcrime = pf.feols(formula_Wcrime, data=boot_df)

        boot_df["resid_lag0"] = model1_lag0.resid
        boot_df["resid_lag1"] = model1_lag1.resid
        boot_df["resid_Wlag0"] = model1_Wlag0.resid
        boot_df["resid_Wlag1"] = model1_Wlag1.resid
        boot_df["Wcrime_resid"] = model1_Wcrime.resid

        model2 = pf.feols(formula2, data=boot_df)

        coefs = model2.coef().to_dict()

    except Exception as e:
        print(f"Bootstrap {b} failed: {e}")
        coefs = None

    del boot_df
    gc.collect()

    return coefs


def full_2stage_cluster_bootstrap(
    df,
    formula1_lag0, formula1_lag1,
    formula1_Wlag0, formula1_Wlag1,
    formula_Wcrime,
    formula2,
    original_coefs,
    footfall_factor,
    B, seed=42, n_jobs=-1
):
    """Parallelized bootstrap using joblib"""

    # pre-split data
    grouped = {k: v for k, v in df.groupby("hex_id", sort=False)}
    clusters = np.array(list(grouped.keys()))

    print("Number of clusters:", len(clusters))

    boot_coefs_list = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(single_bootstrap_iteration)(
            b, grouped, clusters,
            formula1_lag0, formula1_lag1,
            formula1_Wlag0, formula1_Wlag1,
            formula_Wcrime, formula2, seed
        ) for b in range(B)
    )

    # filter out failed iterations
    boot_coefs_list = [c for c in boot_coefs_list if c is not None]

    if len(boot_coefs_list) == 0:
        raise ValueError("All bootstrap iterations failed!")
    else:
        print(f"Valid bootstrap draws: {len(boot_coefs_list)} / {B}")

    boot_df = pd.DataFrame(boot_coefs_list)
    boot_se = boot_df.std(ddof=1)
    boot_p_v2 = boot_df.apply(lambda x: 2 * min((x <= 0).mean(), (x > 0).mean())) # percentile-based bootstrap p-value
    boot_df_exp = np.exp(boot_df * footfall_factor)
    ci = boot_df_exp.quantile([0.025, 0.975])

    del boot_df
    gc.collect()

    return boot_se, boot_p_v2, ci

def load_city_data(city_folder, crime_types):
    df_footfall = pd.read_csv(f"../preprocessing/mobility_data/final_data/{city_folder}_mobility_footfall_hex.csv", index_col=0)

    dfs = []
    for crime in crime_types:
        df_crime = pd.read_csv(f"../preprocessing/crime_data/{city_folder}_{crime}_all_final_hex.csv", index_col=0)
        dfs.append(df_crime)

    df_crimes = sum(dfs)

    hex_df = pd.read_csv(f'../preprocessing/hexagons_list/{city_folder}_hex_list.csv', index_col=0)
    original_hex_order = hex_df.index.tolist()

    df_crimes = df_crimes.loc[:, df_crimes.columns.isin(hex_df.index)]

    df_footfall = df_footfall.reindex(columns=hex_df.index, fill_value=0).fillna(0)
    df_crimes = df_crimes.reindex(columns=hex_df.index, fill_value=0).fillna(0)

    df_footfall = make_datetime_index(df_footfall)
    df_crimes = make_datetime_index(df_crimes)

    return df_footfall, df_crimes, original_hex_order


def make_datetime_index(df):
    df = df.reset_index().rename(columns={"index": "time_key"})
    df[["hour_of_year", "year"]] = df["time_key"].str.split("_", expand=True)
    df["hour_of_year"] = df["hour_of_year"].astype(int)
    df["year"] = df["year"].astype(int)
    df["datetime"] = pd.to_datetime(df["year"].astype(str) + "-01-01") + \
                     pd.to_timedelta(df["hour_of_year"], unit="h")
    return df.set_index("datetime").drop(columns=["time_key", "hour_of_year", "year"])


def expand_daily_to_hourly(df_cov):
    df_cov = df_cov.copy().astype(int)
    df_cov.index = pd.to_datetime(df_cov.index)
    return df_cov.reindex(df_cov.index.repeat(24)).assign(
        datetime=lambda x: np.repeat(df_cov.index, 24) +
                            pd.to_timedelta(np.tile(range(24), len(df_cov)), unit="h")
    ).set_index("datetime")


def load_covariates(city):
    df_sports = pd.read_csv(
        f"../confounders_and_moderators_datasets/{city}_sport_events.csv",
        index_col=0, parse_dates=True
    )
    df_holidays = pd.read_csv(
        f"./confounders_and_moderators_datasets/Holidays_{city}.csv",
        index_col=0, parse_dates=True
    )
    df_extras = pd.read_csv(
        f"./confounders_and_moderators_datasets/{city}_extra_events.csv",
        index_col=0, parse_dates=True
    )
    return expand_daily_to_hourly(df_sports), expand_daily_to_hourly(df_holidays), expand_daily_to_hourly(df_extras)

def generate_regression_results_model5(lags, cities, model_name, crime_types, covariates,footfall_factor, n_jobs=-1, B=500):

    for city in cities:
        print(f"\n=== CITY: {city} ===")
        city_results = []

        for crime in crime_types:
            print(f"\nDoing CRIME={crime}...")

            W = np.load(f"../preprocessing/W_matrix/{city}_W_matrix.npy")

            df_footfall, df_crime, original_hex_order = load_city_data(city_folder=city, crime_types=crime)
            df_crime = df_crime.astype(np.float32)
            df_footfall = df_footfall.astype(np.float32)

            hexes_with_crime = df_crime.columns[df_crime.sum(axis=0) > 0]
            df_crime = df_crime[hexes_with_crime]
            df_footfall = df_footfall[hexes_with_crime]

            # verify ordering consistency
            assert list(df_crime.columns) == list(hexes_with_crime), "Column ordering mismatch!"

            # get the indices of the kept hexagons in the original ordering
            keep_indices = [original_hex_order.index(h) for h in hexes_with_crime]
            # subset W matrix to only include rows/columns for kept hexagons
            W = W[np.ix_(keep_indices, keep_indices)]
            assert W.shape[0] == len(hexes_with_crime), "W matrix dimension mismatch!"

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

            crime_long = None
            footfall_long = None
            gc.collect()

            df_sports_hr, df_holidays_hr, df_extras_hr = load_covariates(city)
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

            # lags & spatial lags
            for k in lags + [lags[-1] + 1, lags[-1] + 2]:
                df[f'footfall_lag{k}'] = df.groupby("hex_id")["footfall"].shift(k)
                ff_mat = df_footfall.shift(k).to_numpy()
                W_ff = W @ ff_mat.T
                df[f'W_footfall_lag{k}'] = W_ff.T.flatten(order='C') # necessary to put back into long format

            for k in [1, 2, 3]:
                crime_mat = df_crime.shift(k).to_numpy()
                W_crime = W @ crime_mat.T
                df[f"W_crime_lag{k}"] = W_crime.T.flatten(order='C')

            del df_crime, df_footfall
            gc.collect()

            df.dropna(inplace=True)

            # set to category bc they use less memory
            df["hour_of_day"] = df["datetime"].dt.hour.astype("category")
            df["day_of_week"] = df["datetime"].dt.weekday.astype("category")
            df["month"] = df["datetime"].dt.month.astype("category")
            df["year"] = df["datetime"].dt.year.astype("category")

            fe_str = " + ".join(["hex_id", "hour_of_day", "day_of_week", "month", "year"])

            exog_covs = ["footfall_lag2", "footfall_lag3", "W_footfall_lag2", "W_footfall_lag3"] + covariates
            instruments = ["footfall_lag4", "footfall_lag5"]
            W_instruments = ["W_footfall_lag4", "W_footfall_lag5"]
            Crime_instruments = ["W_crime_lag2", "W_crime_lag3"]

            formula1_lag0 = "footfall_lag0 ~ " + " + ".join(instruments + exog_covs) + " | " + fe_str
            formula1_lag1 = "footfall_lag1 ~ " + " + ".join(instruments + exog_covs) + " | " + fe_str
            formula1_Wlag0 = "W_footfall_lag0 ~ " + " + ".join(W_instruments + exog_covs) + " | " + fe_str
            formula1_Wlag1 = "W_footfall_lag1 ~ " + " + ".join(W_instruments + exog_covs) + " | " + fe_str
            formula_Wcrime = "W_crime_lag1 ~ " + " + ".join(Crime_instruments + exog_covs) + " | " + fe_str

            # original stage 1 of 2SRI
            model1_lag0 = pf.feols(formula1_lag0, data=df, vcov={"CRV1": "hex_id"})
            model1_lag1 = pf.feols(formula1_lag1, data=df, vcov={"CRV1": "hex_id"})
            model1_Wlag0 = pf.feols(formula1_Wlag0, data=df, vcov={"CRV1": "hex_id"})
            model1_Wlag1 = pf.feols(formula1_Wlag1, data=df, vcov={"CRV1": "hex_id"})
            model1_WCrime = pf.feols(formula_Wcrime, data=df, vcov={"CRV1": "hex_id"})

            df["footfall_resid_lag0"] = model1_lag0.resid()
            df["footfall_resid_lag1"] = model1_lag1.resid()
            df["W_footfall_resid_lag0"] = model1_Wlag0.resid()
            df["W_footfall_resid_lag1"] = model1_Wlag1.resid()
            df["Wcrime_resid"] = model1_WCrime.resid()

            del model1_lag0, model1_lag1, model1_Wlag0, model1_Wlag1, model1_WCrime
            gc.collect()

            regressors = (
                [f"footfall_lag{k}" for k in lags] +
                [f"W_footfall_lag{k}" for k in lags] +
                ["W_crime_lag1"] +
                covariates +
                [
                    'footfall_resid_lag0', 'footfall_resid_lag1',
                    'W_footfall_resid_lag0', 'W_footfall_resid_lag1',
                    'Wcrime_resid'
                ]
            )

            formula2 = "crime ~ " + " + ".join(regressors) + " | " + fe_str

            # original stage 2 of 2SRI
            model2 = pf.fepois(formula2, data=df, vcov={"CRV1": "hex_id"})
            print(model2.summary())

            # get original coefficients from boostrap
            original_coefs = model2.coef()

            # parallelized boostrap
            print("Running PARALLELIZED 2-stage cluster bootstrap...")
            boot_se, boot_p_v2, ci = full_2stage_cluster_bootstrap(
                df=df,
                formula1_lag0=formula1_lag0,
                formula1_lag1=formula1_lag1,
                formula1_Wlag0=formula1_Wlag0,
                formula1_Wlag1=formula1_Wlag1,
                formula_Wcrime=formula_Wcrime,
                formula2=formula2,
                original_coefs=original_coefs,
                footfall_factor=footfall_factor,
                B=B,
                n_jobs=n_jobs
            )
            print("CI: ",ci)

            # result estraction
            crime_str = crime[0] if len(crime) == 1 else 'All'
            results_df = model2.tidy()

            for idx, row in results_df.iterrows():
                var = idx
                beta = row['Estimate']
                std_err = row['Std. Error']
                p_val = row['Pr(>|t|)']
                std_err_boot = boot_se.get(var, np.nan)
                #p_val_boot = boot_p.get(var, np.nan)
                p_val_boot_v2 = boot_p_v2.get(var, np.nan)
                exp_beta_factor = np.exp(beta * footfall_factor)
                ci_low = ci.loc[0.025].get(var, np.nan)
                ci_high = ci.loc[0.975].get(var, np.nan)

                city_results.append({
                    "crime_type": crime_str,
                    "variable": var,
                    "beta": beta,
                    "exp_beta_factor": exp_beta_factor,
                    "std_err_boot": std_err_boot,
                    "exp_std_err_factor_boot": exp_beta_factor * footfall_factor * std_err_boot,
                    #"p_value_boot": p_val_boot,
                    "p_value_boot_v2": p_val_boot_v2,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "std_err": std_err,
                    "p_value": f"{p_val:.8f}",
                    "num_obs": int(model2._N),
                    "log-likelihood": model2._loglik,
                    "pearson_chi2": model2._pearson_chi2,
                    "pseudo_r2": model2._pseudo_r2,
                    "B": B
                })

            del df, model2, original_coefs, boot_se #, boot_p
            gc.collect()

            city_results_df = pd.DataFrame(city_results)
            city_results_df.to_csv(
                f"results/Model_with_2SRI/{city}_Pfixest_{model_name}_{crime_str}_1H_N{n_jobs}_B{B}_FF{footfall_factor}.csv",
                index=False
            )

            print(f"Saved {city} results!")

if __name__ == "__main__":
    n_jobs = 12
    B = 500

    print(f"Running with n_jobs={n_jobs}, B={B}")

    generate_regression_results_model5(
        lags=[0, 1, 2, 3],
        cities=['Chicago'],
        model_name='bootstrap_parallel',
        crime_types = [
                        ["Burglary"],
                        ["Motor Vehicle Theft"],
                        ["Assault"],
                        ["Homicide"],
                        ["Robbery"],
                        ["Burglary", "Assault", "Robbery", "Homicide", "Motor Vehicle Theft"]
                    ],
        covariates=['extra_event', 'holiday', 'sport_event'],
        footfall_factor=7000,
        n_jobs=n_jobs,
        B=B
    )
