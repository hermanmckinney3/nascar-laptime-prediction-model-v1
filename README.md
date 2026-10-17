# NASCAR Next-Lap Pace Prediction (V1)

Predicting a NASCAR Cup Series driver's next green-flag lap time from live race data, using lap-by-lap timing, pit stop, and caution data from Michigan International Speedway (2021–2026).

## Results

| Method | Test MAE (seconds) |
|---|---|
| **XGBoost model** | **0.166** |
| Current lap (next lap = this lap) | 0.189 |
| Rolling 3-lap average | 0.207 |
| Rolling 5-lap average | 0.235 |

The model beats the strongest simple baseline by about **12%**, trained on 2021–2024 and tested on unseen 2025–2026 races.

## Approach
- **Feature engineering:** right- and left-side tire age, green-flag "runs" that reset at tire changes and restarts, rolling pace and pace trend (tire falloff) within each run, and pace relative to the field
- **Target:** the *change* from the current lap to the next lap, rather than the raw lap time
- **Error analysis:** the largest misses clustered on laps just before cautions and at stage ends; those laps were excluded as outside the model's scope (clean green-flag racing)

## How to Run
1. Download or clone this repository
2. `pip install -r requirements.txt`
3. Open `nascar_next_lap_v1.ipynb` and run all cells

Raw data was pulled with pynascar (see `fetch_data.ipynb`); the CSV files are included in `data/`, so no API access is needed.

## Limitations
- Single track (Michigan); four training races
- Accuracy is measured on laps that stayed green. In live use, laps leading into a caution would be included and reduce accuracy.

## Next Steps
- V2: predict each driver's pace over the next 5 laps, relative to the field
- A pit-or-stay-out strategy simulation built on V2
