import pandas as pd
import numpy as np
import pyfixest as pf
from joblib import Parallel, delayed
import gc
from scipy.stats import norm
import argparse

# parallelized cluster bootstrap function
def single_bootstrap_iteration(
    b, df, clusters,
    formula1_lag0,
    formula1_Wlag0,
    formula_Wcrime,
    formula1_interaction,
    formula2,
    seed
):
    # unique seed per iteration
    np.random.seed(seed + b) 

    try:
        # sample clusters with replacement
        sampled_clusters = np.random.choice(clusters, size=len(clusters), replace=True)
        boot_df = pd.concat([df[df["hex_id"] == c] for c in sampled_clusters], axis=0, ignore_index=True)

        m1_lag0 = pf.feols(formula1_lag0, data=boot_df)
        m1_Wlag0 = pf.feols(formula1_Wlag0, data=boot_df)
        m1_Wcrime = pf.feols(formula_Wcrime, data=boot_df)
        m1_interaction = pf.feols(formula1_interaction, data=boot_df)

        boot_df["footfall_resid_lag0"] = m1_lag0.resid()
        boot_df["W_footfall_resid_lag0"] = m1_Wlag0.resid()
        boot_df["Wcrime_resid"] = m1_Wcrime.resid()
        boot_df["footfall_interaction_resid"] = m1_interaction.resid()

        del m1_lag0, m1_Wlag0, m1_Wcrime, m1_interaction
        gc.collect()

        m2 = pf.fepois(formula2, data=boot_df)

        del boot_df
        gc.collect()

        return m2.coef().to_dict()

    except Exception as e:
        print(f"Bootstrap iteration {b} failed: {str(e)}")
        return None


def full_2stage_cluster_bootstrap(
    df,
    formula1_lag0,
    formula1_Wlag0,
    formula_Wcrime,
    formula1_interaction,
    formula2,
    original_coefs,
    footfall_factor,
    B, seed=42, n_jobs=-1
):
    """Parallelized bootstrap using joblib"""

    clusters = df["hex_id"].unique()

    print(f"Running {B} bootstrap iterations with {n_jobs} jobs...")
    boot_coefs_list = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(single_bootstrap_iteration)(
            b, df, clusters,
            formula1_lag0,
            formula1_Wlag0,
            formula_Wcrime,
            formula1_interaction,
            formula2, seed
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
    t_stats = original_coefs / boot_se
    boot_p = 2 * (1 - norm.cdf(np.abs(t_stats)))
    boot_p = pd.Series(boot_p, index=boot_se.index)
    boot_p_v2 = boot_df.apply(lambda x: 2 * min((x <= 0).mean(), (x > 0).mean()))
    boot_df_exp = np.exp(boot_df * footfall_factor)  # transform coefficients to IRR scale
    ci = boot_df_exp.quantile([0.025, 0.975])

    return boot_se, boot_p, boot_p_v2, ci, boot_df

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
    df["datetime"] = df["datetime"].dt.floor("6H")

    return (df.drop(columns=["time_key", "hour_of_year", "year"]).groupby("datetime", as_index=True).sum())


def expand_daily_to_6h(df_cov):
    df_cov = df_cov.copy().astype(int)
    df_cov.index = pd.to_datetime(df_cov.index)

    return df_cov.reindex(df_cov.index.repeat(4)).assign(
            datetime=lambda x: np.repeat(df_cov.index, 4) + pd.to_timedelta(np.tile([0, 6, 12, 18], len(df_cov)), unit="h")
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

def generate_regression_results_heterogeneity(
    lags, cities, model_name, crime_types, covariates, file_name_het,
    heterogeneity_var, footfall_factor, n_jobs=-1, B=500
):
    """
    Run 2SRI model with heterogeneity analysis based on a sociodemographic variable.

    Args:
        lags: List of lag values to include (e.g., [0, 1, 2])
        cities: List of city names
        model_name: Name for output files
        crime_types: List of crime type lists
        covariates: List of covariate names
        heterogeneity_var: Column name from sociodemographic file (e.g., 'Median_household_income')
        n_jobs: Number of parallel jobs for bootstrap
        B: Number of bootstrap iterations
    """

    for city in cities:
        print(f"\n{'='*60}")
        print(f"=== CITY: {city} ===")
        print(f"=== HETEROGENEITY VAR: {heterogeneity_var} ===")
        print(f"{'='*60}")

        # load sociodemographic data
        df_heterogeneity = pd.read_csv(
            f"../preprocessing/confounders_and_moderators_datasets/{city}_{file_name_het}",
            index_col=0
        )
        # ensure hex_index is the index for mapping
        df_heterogeneity = df_heterogeneity.set_index('hex_index')

        print(f"Sociodemographic data loaded: {len(df_heterogeneity)} hexagons")
        print(f"{heterogeneity_var} range: {df_heterogeneity[heterogeneity_var].min():.2f} - {df_heterogeneity[heterogeneity_var].max():.2f}")
        print(f"{heterogeneity_var} median: {df_heterogeneity[heterogeneity_var].median():.2f}")

        for crime in crime_types:
            city_results = []
            crime_str = crime[0] if len(crime) == 1 else 'All'
            print(f"\n--- CRIME: {crime_str} ---")

            W = np.load(f"../preprocessing/W_matrix/{city}_W_matrix.npy")

            df_footfall, df_crime, original_hex_order = load_city_data(city_folder=city, crime_types=crime)
            df_crime = df_crime.astype(np.float32)
            df_footfall = df_footfall.astype(np.float32)

            hexes_with_crime = df_crime.columns[df_crime.sum(axis=0) > 0]
            df_crime = df_crime[hexes_with_crime]
            df_footfall = df_footfall[hexes_with_crime]

            assert list(df_crime.columns) == list(hexes_with_crime), "Column ordering mismatch!"

            keep_indices = [original_hex_order.index(h) for h in hexes_with_crime]
            W = W[np.ix_(keep_indices, keep_indices)]
            assert W.shape[0] == len(hexes_with_crime), "W matrix dimension mismatch!"

            # create long format
            crime_long = df_crime.reset_index().melt(id_vars="datetime", var_name="hex_id", value_name="crime")
            footfall_long = df_footfall.reset_index().melt(id_vars="datetime", var_name="hex_id", value_name="footfall")
            df = pd.merge(crime_long, footfall_long, on=["datetime", "hex_id"], how="outer")

            # apply sample correction
            SAMPLE_CORRECTIONS = {
                "Baltimore": 28.51,
                "Chicago": 36.41,
                "Philadelphia": 39.75,
            }
            df["footfall"] = df["footfall"] * SAMPLE_CORRECTIONS[city]

            del crime_long, footfall_long
            gc.collect()

            # load covariates
            df_sports_hr, df_holidays_hr, df_extras_hr = load_covariates(city)
            df = (df.merge(df_sports_hr, on="datetime", how="left")
                    .merge(df_holidays_hr, on="datetime", how="left")
                    .merge(df_extras_hr, on="datetime", how="left"))
            df.rename(columns={"Event": "extra_event", "Sport_event": "sport_event", "is_holiday": "holiday"}, inplace=True)

            # merge sociodemographic variables and create groups
            df = df.merge(
                df_heterogeneity[[heterogeneity_var]].reset_index().rename(columns={"hex_index": "hex_id"}),
                on="hex_id",
                how="left"
            )

            # create binary group indicator (median split)
            # calculate median at hex level (not observation level)
            hex_level_values = df.groupby("hex_id")[heterogeneity_var].first()
            median_val = hex_level_values.median()

            # map back to full dataframe
            hex_to_group = (hex_level_values >= median_val).astype(int)
            df["high_group"] = df["hex_id"].map(hex_to_group)

            print(f"\nGroup split at median = {median_val:.2f}")
            print(f"Low group (high_group=0): {(hex_to_group == 0).sum()} hexagons")
            print(f"High group (high_group=1): {(hex_to_group == 1).sum()} hexagons")

            # sort and verify ordering
            hex_order = df_footfall.columns.tolist()
            df["hex_id"] = pd.Categorical(df["hex_id"], categories=hex_order, ordered=True)
            df = df.sort_values(["datetime", "hex_id"]).reset_index(drop=True)
            df["hex_id"] = df["hex_id"].astype(str)

            expected_order = pd.MultiIndex.from_product([df_footfall.index, df_footfall.columns], names=['datetime', 'hex_id'])
            actual_order = pd.MultiIndex.from_arrays([df['datetime'], df['hex_id']])
            assert expected_order.equals(actual_order), "Ordering mismatch for spatial lag assignment!"

            for k in lags + [lags[-1] + 1, lags[-1] + 2]:
                df[f'footfall_lag{k}'] = df.groupby("hex_id")["footfall"].shift(k)
                ff_mat = df_footfall.shift(k).to_numpy()
                W_ff = W @ ff_mat.T
                df[f'W_footfall_lag{k}'] = W_ff.T.flatten(order='C')

            for k in [1, 2, 3]:
                crime_mat = df_crime.shift(k).to_numpy()
                W_crime = W @ crime_mat.T
                df[f"W_crime_lag{k}"] = W_crime.T.flatten(order='C')

            del df_crime, df_footfall
            gc.collect()
            
            # interaction for footfall_lag0 only
            df["footfall_lag0_X_high"] = df["footfall_lag0"] * df["high_group"]

            # interactions for instruments
            for k in [lags[-1] + 1, lags[-1] + 2]:  # lag3, lag4
                df[f"footfall_lag{k}_X_high"] = df[f"footfall_lag{k}"] * df["high_group"]

            df.dropna(inplace=True)

            df["six_hour_block"] = df["datetime"].dt.hour // 6
            # set to category bc they use less memory
            df["day_of_week"] = df["datetime"].dt.weekday.astype("category")
            df["month"] = df["datetime"].dt.month.astype("category")
            df["year"] = df["datetime"].dt.year.astype("category")

            # build formulas with interactions
            fe_str = " + ".join(["hex_id", "six_hour_block", "day_of_week", "month", "year"])

            # exogenous covariates
            exog_covs = (
                ["footfall_lag1", "footfall_lag2", "W_footfall_lag1", "W_footfall_lag2"] +
                covariates
            )

            # instruments for footfall_lag0
            instruments = ["footfall_lag3", "footfall_lag4"]

            # instruments for W_footfall_lag0
            W_instruments = ["W_footfall_lag3", "W_footfall_lag4"]

            # instruments for interaction term (footfall_lag0_X_high)
            interaction_instruments = ["footfall_lag3_X_high", "footfall_lag4_X_high"]

            # instruments for W_crime
            crime_instruments = ["W_crime_lag2", "W_crime_lag3"]

            # stage 1 formulas
            formula1_lag0 = "footfall_lag0 ~ " + " + ".join(instruments + exog_covs) + " | " + fe_str
            formula1_Wlag0 = "W_footfall_lag0 ~ " + " + ".join(W_instruments + exog_covs) + " | " + fe_str
            formula_Wcrime = "W_crime_lag1 ~ " + " + ".join(crime_instruments + exog_covs) + " | " + fe_str

            # stage 1 for the interaction term
            formula1_interaction = ("footfall_lag0_X_high ~ " + " + ".join(interaction_instruments + exog_covs) + " | " + fe_str)

            # original stage 1 for 2SRI
            print("\nRunning Stage 1 models...")

            model1_lag0 = pf.feols(formula1_lag0, data=df, vcov={"CRV1": "hex_id"})
            model1_Wlag0 = pf.feols(formula1_Wlag0, data=df, vcov={"CRV1": "hex_id"})
            model1_WCrime = pf.feols(formula_Wcrime, data=df, vcov={"CRV1": "hex_id"})
            model1_interaction = pf.feols(formula1_interaction, data=df, vcov={"CRV1": "hex_id"})

            df["footfall_resid_lag0"] = model1_lag0.resid()
            df["W_footfall_resid_lag0"] = model1_Wlag0.resid()
            df["Wcrime_resid"] = model1_WCrime.resid()
            df["footfall_interaction_resid"] = model1_interaction.resid()

            del model1_lag0, model1_Wlag0, model1_WCrime, model1_interaction
            gc.collect()

            # stage 2 with interactions (only footfall_lag0)
            regressors = (
                # main effects
                [f"footfall_lag{k}" for k in lags] +
                [f"W_footfall_lag{k}" for k in lags] +
                # interaction effect (ONLY lag0)
                ["footfall_lag0_X_high"] +
                # other controls
                ["W_crime_lag1"] +
                covariates +
                # control function residuals
                ['footfall_resid_lag0', 'W_footfall_resid_lag0', 'Wcrime_resid', 'footfall_interaction_resid']
            )

            formula2 = "crime ~ " + " + ".join(regressors) + " | " + fe_str

            print("\nRunning Stage 2 model...")
            model2 = pf.fepois(formula2, data=df, vcov={"CRV1": "hex_id"})
            print(model2.summary())

            original_coefs = model2.coef()

            # parallelized bootstrap
            print("\nRunning PARALLELIZED 2-stage cluster bootstrap...")
            boot_se, boot_p, boot_p_v2, ci, boot_df = full_2stage_cluster_bootstrap(
                df=df,
                formula1_lag0=formula1_lag0,
                formula1_Wlag0=formula1_Wlag0,
                formula_Wcrime=formula_Wcrime,
                formula1_interaction=formula1_interaction,
                formula2=formula2,
                original_coefs=original_coefs,
                footfall_factor=footfall_factor,
                B=B,
                n_jobs=n_jobs
            )

            # compute high-group total effect p-value from bootstrap distribution
            boot_high_total = boot_df["footfall_lag0"] + boot_df["footfall_lag0_X_high"]
            # original combined coefficient
            orig_high_total = original_coefs["footfall_lag0"] + original_coefs["footfall_lag0_X_high"]
            # bootstrap SE and p-values for the combined effect
            high_total_se = boot_high_total.std(ddof=1)
            high_total_p_v2 = 2 * min((boot_high_total <= 0).mean(), (boot_high_total > 0).mean())
            # CI on IRR scale
            boot_high_total_exp = np.exp(boot_high_total * footfall_factor)
            high_total_ci_low = boot_high_total_exp.quantile(0.025)
            high_total_ci_high = boot_high_total_exp.quantile(0.975)

            del boot_df
            gc.collect()

            # results extraction with heterogeneity info
            results_df = model2.tidy()
            for idx, row in results_df.iterrows():
                var = idx
                beta = row['Estimate']
                std_err = row['Std. Error']
                p_val = row['Pr(>|t|)']
                std_err_boot = boot_se.get(var, np.nan)
                p_val_boot = boot_p.get(var, np.nan)
                p_val_boot_v2 = boot_p_v2.get(var, np.nan)
                exp_beta_factor = np.exp(beta * footfall_factor)
                ci_low = ci.loc[0.025].get(var, np.nan)
                ci_high = ci.loc[0.975].get(var, np.nan)

                city_results.append({
                    "crime_type": crime_str,
                    "heterogeneity_var": heterogeneity_var,
                    "variable": var,
                    "beta": beta,
                    "exp_beta_factor": exp_beta_factor,
                    "std_err_boot": std_err_boot,
                    "exp_std_err_factor_boot": exp_beta_factor * footfall_factor * std_err_boot,
                    "p_value_boot": p_val_boot,
                    "p_value_boot_v2": p_val_boot_v2,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "std_err": std_err,
                    "p_value": f"{p_val:.8f}",
                    "num_obs": int(model2._N),
                    "log-likelihood": model2._loglik,
                    "pearson_chi2": model2._pearson_chi2,
                    "pseudo_r2": model2._pseudo_r2,
                    "B": B,
                    "median_split_value": median_val,
                    "n_hexagons_low": (hex_to_group == 0).sum(),
                    "n_hexagons_high": (hex_to_group == 1).sum()
                })

            # add high-group total effect row
            city_results.append({
                "crime_type": crime_str,
                "heterogeneity_var": heterogeneity_var,
                "variable": "footfall_lag0_HIGH_TOTAL",
                "beta": orig_high_total,
                "exp_beta_factor": np.exp(orig_high_total * footfall_factor),
                "std_err_boot": high_total_se,
                "exp_std_err_factor_boot": np.exp(orig_high_total * footfall_factor) * footfall_factor * high_total_se,
                "p_value_boot": np.nan,
                "p_value_boot_v2": high_total_p_v2,
                "ci_low": high_total_ci_low,
                "ci_high": high_total_ci_high,
                "std_err": np.nan,
                "p_value": np.nan,
                "num_obs": int(model2._N),
                "log-likelihood": model2._loglik,
                "pearson_chi2": model2._pearson_chi2,
                "pseudo_r2": model2._pseudo_r2,
                "B": B,
                "median_split_value": median_val,
                "n_hexagons_low": (hex_to_group == 0).sum(),
                "n_hexagons_high": (hex_to_group == 1).sum()
            })

            del df, model2, original_coefs, boot_se, boot_p
            gc.collect()

            city_results_df = pd.DataFrame(city_results)
            city_results_df.to_csv(
                f"results/Model_with_2SRI/{city}_Pfixest_{model_name}_{crime_str}_heterogeneity_{heterogeneity_var}_6H_N{n_jobs}_B{B}.csv",
                index=False
            )

            print(f"Saved {city} + {crime_str} + {heterogeneity_var} results!")

            print("\n" + "="*60)
            print("INTERPRETATION GUIDE:")
            print("="*60)
            print(f"• footfall_lag0: Effect in LOW {heterogeneity_var} hexagons (reference)")
            print(f"• footfall_lag0_X_high: ADDITIONAL effect in HIGH {heterogeneity_var} hexagons")
            print(f"• Total effect in HIGH group = footfall_lag0 + footfall_lag0_X_high")
            print(f"\nIf footfall_lag0_X_high is significant:")
            print(f"  → Positive: Effect is STRONGER in high {heterogeneity_var} areas")
            print(f"  → Negative: Effect is WEAKER in high {heterogeneity_var} areas")
            print("="*60)

if __name__ == "__main__":
    n_jobs = 25
    B = 500
    print(f"Running with n_jobs={n_jobs}, B={B}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--cities', type=str, nargs='+', required=True)
    parser.add_argument('--crime_types', type=str, nargs='+', required=True)

    args = parser.parse_args()
    print(f"Running with cities={args.cities} and crime_types={args.crime_types}")
    # put the corresponding variable given the file name
    for het_var in ["Percent_single_parent_houshold"]: #  "poi_diversity","night_businesses_proportion?", "Median_household_income", "Percent_occupied_housing_units", "Percent_unemployed", "Ethnic_diversity"
        generate_regression_results_heterogeneity(
            lags=[0, 1, 2],
            cities=args.cities,
            model_name='bootstrap_heterogeneity',
            crime_types=[args.crime_types],
            covariates=['extra_event', 'holiday', 'sport_event'],
            file_name_het='sociodem_hex.csv', #  mobility_hex.csv, sociodem_hex.csv
            heterogeneity_var=het_var,
            footfall_factor=36000,
            n_jobs=n_jobs,
            B=B
        )
