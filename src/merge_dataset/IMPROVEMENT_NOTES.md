# Weather Forecast Deduplication Improvement

## Problem

When loading HRRR weather forecasts, there are massive overlaps between consecutive forecast files:
- Each forecast file covers 48 hours (f00-f48)
- Files are issued daily
- Result: ~25 hours of overlap between consecutive days
- For the same target datetime, multiple forecasts exist with different horizons

Example:
```
Target datetime: 2024-01-02 10:00 AM

Forecast 1 (issued Jan 1 00:00):
  - forecast_hour: f34 (34 hours ahead)
  - Less accurate (long horizon)

Forecast 2 (issued Jan 2 00:00):
  - forecast_hour: f10 (10 hours ahead)
  - More accurate (short horizon)
```

## Old Approach

```python
# Keep FIRST forecast (by file order = earliest issued)
df_weather.drop_duplicates(subset=['datetime', 'zone'], keep='first')
```

**Problem**: Keeps longer-horizon (less accurate) forecasts
- Average forecast horizon: f30.2
- Uses forecasts that were made 30+ hours in advance

## New Approach (Implemented)

```python
# Sort by forecast_hour, then keep first (shortest horizon)
df_weather = df_weather.sort_values('forecast_hour')
df_weather = df_weather.drop_duplicates(subset=['datetime', 'zone'], keep='first')
```

**Benefit**: Keeps shorter-horizon (more accurate) forecasts
- Average forecast horizon: f17.8
- **12.4 hour improvement** on average
- Uses the most recent forecast available for each datetime

## Impact

### Quantitative
- **Same number of rows** (deduplication still works correctly)
- **Better forecast quality**: 12.4 hours closer to target time on average
- **No data leakage**: Still uses only forecasts available at that time

### Forecast Horizon Distribution

**Old method:**
- Min: f00, Max: f48
- Mean: f30.2, Median: f32
- Biased toward long-horizon forecasts

**New method:**
- Min: f00, Max: f48
- Mean: f17.8, Median: f16
- Biased toward short-horizon (more accurate) forecasts

## Why This Matters

Weather forecast accuracy degrades with forecast horizon:
- f00-f06: Very accurate (0-6 hours ahead)
- f07-f24: Good accuracy (same day)
- f25-f48: Lower accuracy (1-2 days ahead)

By preferring shorter horizons, the model trains on more realistic weather data that better represents what would be available in production trading scenarios.

## Code Changes

**File**: `src/merge_dataset/loader.py`

**Lines 121-135**: Updated weather deduplication logic
- Added sort by `forecast_hour` before deduplication
- Updated comments to reflect new strategy
- Added verbose output showing improvement

## Testing

See `test_improved_dedup.py` for validation:
- Compares old vs new method on 3 days of 2024 data
- Shows 12.4 hour average improvement
- Confirms same row count (no bugs in deduplication)

## Next Steps

After rebuilding cache with new logic:
1. Retrain baseline models with improved weather data
2. Compare model performance (expect slight improvement)
3. Consider adding `forecast_hour` as a feature to help model learn forecast uncertainty
