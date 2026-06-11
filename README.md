# ambient-population-crime-causal
This repository contains all the code used for the paper "Ambient Population and Crime: The Fragility of Causal Links in Urban Environments" by Ariadna Albors Zumel, Michele Tizzoni, Wilson Hernández, and Gian Maria Campedelli.

📄 You can find the full paper here:

## Introduction
The relationship between ambient population and crime has long been studied, with foundational theories offering competing mechanisms: increased pedestrian activity may enhance guardianship and informal surveillance, but also generate opportunities by increasing suitable targets. Yet, nearly all empirical evidence remains correlational, despite causal identification being essential for policy. We causally address this link using high-resolution smartphone-derived footfall data from Baltimore, Chicago, and Philadelphia (June 2023–June 2025), estimating a pipeline of increasingly rigorous models—from naïve baselines to fully instrumented two-stage residual inclusion specifications with distributed lags, two-way fixed effects, and spatial spillovers. The results are sobering: associations between ambient population and crime progressively vanish as endogeneity is addressed, and none survive correction for multiple testing. The few effects that do emerge are sensitive to crime type, temporal granularity, and neighborhood context. Our findings caution against universal claims linking foot traffic to crime and call for locally tailored, causally informed policy design.

## Code structure

### Preprocessing
- crime_data:
  1. Download the data for each city ([Baltimore](https://data.baltimorecity.gov/datasets/baltimore::nibrs-group-a-crime-data/about), [Chicago](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/about_data), and [Philadelphia](https://opendataphilly.org/datasets/crime-incidents/)) and put in the folder structure "raw_data/<city_name>".
  2. `1-generate_raw_datasets.py`: It loads the raw crime datasets for Baltimore, Chicago, and Philadelphia, standardizes their formats, filters incidents to the two-year period (30 June 2023–29 June 2025), removes crime records located outside each city's official boundary polygon, and saves the cleaned datasets.
  3. `2-generate_selected_crimes_datasets.py`: It loads the cleaned crime datasets for Baltimore, Chicago, and Philadelphia, selects a common set of violent and property crime types, harmonizes crime labels across cities, and saves the standardized datasets.
  4. `3-generate_crimes_per_hex_dataset.py`: It assigns each crime event to a hexagonal spatial cell, aggregates crime counts by hexagon and hour for each year, and then combines yearly matrices into final spatiotemporal crime grids covering the study period.
- mobility_data:
  1. `0-download_data.py`: It uses deweydatapy package to retrieve metadata for the Weekly Patterns dataset of Advan, identifies all files available between June 2023 and July 2025, and downloads those files to a local directory.
  2. `1-preprocess_mobility_data.py`: It processes weekly Advan mobility pattern files, extracts the POIs located in Baltimore, Chicago, and Philadelphia, converts hourly visit counts into 168 hourly features per week, and saves one cleaned mobility dataset for each study week between June 2023 and June 2025.
  3. `2-get_mobility-data_per_hex.py`: It builds a city-level hexagonal spatial grid, maps the POIs to hex cells, aggregates the weekly Advan visit data into hourly footfall per hexagon, and merges yearly outputs into a unified spatiotemporal mobility dataset for Baltimore, Chicago, and Philadelphia.
- confounders_and_moderators_datasets:
  - `Holidays_<city>.csv`:
  - `<city>_sport_events.csv`:
  - `<city>_extra_events.csv`:
  - `<city>_mobility_hex.csv`:
  - `<city>_sociodem_hex.csv`:
- hexagons_list:
  - `get_final_hex_list.py`: It finds the hexagons present in both the mobility and sociodemographic datasets for each city, reports the mismatches, and saves a CSV of the common hexagons (with their polygon geometry) as the city's final hex list.
  - `<city>_hex_list.csv`: A CSV mapping every retained H3 hexagon (identified by its hex_index) to its boundary geometry as a WKT polygon for each city.
- W_matrix:
  - `make_W_matrix.py`: It loads the final hex list, builds a Queen-contiguity spatial weights matrix over the hexagons (neighbors sharing a border or vertex), row-standardizes it so each row sums to one, and saves the resulting dense W matrix as a .npy file for use in spatial-lag/spillover terms.
  - `<city>_W_matrix.npy`: A dense, row-standardized spatial weights (Queen-contiguity) matrix stored as a NumPy array of shape (n_hexagons × n_hexagons) for each city.

### Models
- `baseline_model_pyfixest_6h.py`: It builds a hexagon–time panel dataset of crime and footfall at 6-hour resolution, and estimates crime models (Negative Binomial or Poisson) to quantify the effect of contemporaneous footfall on crime across different crime types.
- `model2_pyfixest_6h.py`: It builds a hexagon–time panel dataset of crime and footfall at 6-hour resolution, and estimates the expected crime counts using a Poisson model with temporally lagged footfall across different crime types.
- `model3_pyfixest_6h.py`: It builds a hexagon–time panel dataset of crime and footfall at 6-hour resolution, and estimates the expected crime counts using a Poisson model with hexagon fixed-effects, several time fixed effects, three covariates, and the hexagon's footfall lags across different crime types.
- `model4_pyfixest_6h.py`: It builds a hexagon–time panel dataset of crime and footfall at 6-hour resolution, and estimates the expected crime counts using a Poisson model with hexagon fixed-effects, several time fixed effects, three covariates, the hexagon's footfall lags, and spillover effects from neighboring hexagons across different crime counts.
- 2SRI:
  - `model5_pyfixest_1h.py`: It takes Model 4 and uses the 2SRI structure to correct for endogeneity, using 1h granularity.
  - `model5_pyfixest_3h.py`: It takes Model 4 and uses the 2SRI structure to correct for endogeneity, using 3h granularity.
  - `model5_pyfixest_12h.py`: It takes Model 4 and uses the 2SRI structure to correct for endogeneity, using 12h granularity.
  - `model5_pyfixest_6h.py`: It takes Model 4 and uses the 2SRI structure to correct for endogeneity, using 6h granularity.
  - `model5_pyfixest_6h_hex_scaling.py`:  It takes Model 4 and uses the 2SRI structure to correct for endogeneity, using 6h granularity. Moreover, it does a hexagon-level sample correction instead of the city-level sample correction.
  - `model5_pyfixest_6h_heterogeneity.py`: It takes Model 4 and uses the 2SRI structure to correct for endogeneity, using 6h granularity. Moreover, it uses a median split over 7 moderating variables (2 POI-related and 5 sociodemographic) to test heterogeneity.

## Structure of the final causal model using 2SRI (Model 5)
Our causal model is a 2SRI model with fixed effects, covariates, distributed lags, and spillover effects using a modified spatial Durbin term:
![Model equation](model5.jpg)
where $\alpha_i$ is the hexagon fixed effects, $\delta_t$ represents the time fixed effects, $\rho$ is the spatial autoregressive coefficient that measures how strongly neighboring hexagons' past expected log crime counts influence hexagon $i$'s own expected log crime count, the parameter $\mathbf{W}\in \\{0,1\\}^{n\times n}$ is the spatial weights matrix, where $n$ is the total number of hexagons, the coefficient $\beta_0$ is interpreted as the immediate effect of footfall in hexagon $i$ on the log of the expected crime count, the sum $\sum_{k=0}^q\beta_k$ represents the long-run cumulative effect of footfall in hexagon $i$ on the log of the expected crime count, the coefficients $\theta_k$ measure the impact of neighboring footfall on the current hexagon's expected log crime count at lag $k$, the term $\gamma X_{i,t}$ is the time-varying covariates not captured by the fixed effects, and $\hat{u}_{j,i,t}$ are the residuals for the three endogenous variables incorporated as part of the 2SRI model in order to correct for endogeneity.
