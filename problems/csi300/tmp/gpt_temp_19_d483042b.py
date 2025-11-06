import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize output series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Small epsilon to avoid division by zero
    eps = 1e-8
    
    # Required columns
    high = df['high']
    low = df['low']
    open_ = df['open']
    close = df['close']
    volume = df['volume']
    
    # Calculate basic components
    morning_asym = (high - open_) / (close.shift(1) + eps) - (open_ - low) / (close.shift(1) + eps)
    morning_asym *= ((close - low) / (high - low + eps) - 0.5)
    
    afternoon_asym = (close - low) / (close.shift(1) + eps) - (high - close) / (close.shift(1) + eps)
    afternoon_asym *= ((close - low) / (high - low + eps) - 0.5)
    
    # Intraday Transmission Cascade
    intraday_cascade = morning_asym * afternoon_asym * np.sign(morning_asym - afternoon_asym)
    
    # Volume-Volatility components
    vol_morning_asym = volume / (volume.shift(1) + eps) * np.abs(volume - volume.shift(1)) / (volume + eps)
    vol_morning_asym *= (high - low) ** 0.8 * np.sign((close - low) / (high - low + eps) - 0.5)
    
    vol_afternoon_asym = volume / (volume.shift(3) + eps) * np.abs(volume - volume.shift(3)) / (volume + eps)
    vol_afternoon_asym *= (high - low) ** 0.8 * np.sign((close - low) / (high - low + eps) - 0.5)
    
    vol_vol_transmission = vol_morning_asym * vol_afternoon_asym * np.sign(vol_morning_asym - vol_afternoon_asym)
    
    # Synchronized Intraday Transmission
    trans_divergence = np.abs(intraday_cascade - vol_vol_transmission)
    synchronized_intraday = intraday_cascade * vol_vol_transmission * (1 - trans_divergence)
    
    # Persistence components
    morning_persistence = pd.Series(0.0, index=df.index)
    afternoon_persistence = pd.Series(0.0, index=df.index)
    
    for i in range(3, len(df)):
        morning_count_up = 0
        morning_count_down = 0
        afternoon_count_up = 0
        afternoon_count_down = 0
        
        for j in range(i-2, i+1):
            if (high.iloc[j] - open_.iloc[j]) > (open_.iloc[j] - low.iloc[j]):
                morning_count_up += 1
            elif (high.iloc[j] - open_.iloc[j]) < (open_.iloc[j] - low.iloc[j]):
                morning_count_down += 1
                
            if (close.iloc[j] - low.iloc[j]) > (high.iloc[j] - close.iloc[j]):
                afternoon_count_up += 1
            elif (close.iloc[j] - low.iloc[j]) < (high.iloc[j] - close.iloc[j]):
                afternoon_count_down += 1
        
        morning_persistence.iloc[i] = (morning_count_up - morning_count_down) * ((close.iloc[i] - low.iloc[i]) / (high.iloc[i] - low.iloc[i] + eps) - 0.5)
        afternoon_persistence.iloc[i] = (afternoon_count_up - afternoon_count_down) * ((close.iloc[i] - low.iloc[i]) / (high.iloc[i] - low.iloc[i] + eps) - 0.5)
    
    intraday_persistence_cascade = morning_persistence * afternoon_persistence * np.sign(morning_persistence - afternoon_persistence)
    
    # Efficiency components
    morning_efficiency = pd.Series(0.0, index=df.index)
    afternoon_efficiency = pd.Series(0.0, index=df.index)
    
    for i in range(2, len(df)):
        morning_sum = sum(np.abs(high.iloc[j] - open_.iloc[j]) for j in range(i-2, i+1))
        afternoon_sum = sum(np.abs(close.iloc[j] - low.iloc[j]) for j in range(i-2, i+1))
        
        morning_efficiency.iloc[i] = np.abs(high.iloc[i] - open_.iloc[i]) / (morning_sum + eps) * ((close.iloc[i] - low.iloc[i]) / (high.iloc[i] - low.iloc[i] + eps) - 0.5)
        afternoon_efficiency.iloc[i] = np.abs(close.iloc[i] - low.iloc[i]) / (afternoon_sum + eps) * ((close.iloc[i] - low.iloc[i]) / (high.iloc[i] - low.iloc[i] + eps) - 0.5)
    
    intraday_efficiency_ratio = morning_efficiency * afternoon_efficiency
    
    # Volatility components
    morning_true_range = np.maximum(high - low, np.maximum(np.abs(high - open_), np.abs(low - open_)))
    afternoon_true_range = np.maximum(high - low, np.maximum(np.abs(high - close), np.abs(low - close)))
    
    # Calculate volatility stability (10-day rolling std of range/close ratio)
    range_close_ratio = (high - low) / (close + eps)
    vol_stability = (morning_true_range + afternoon_true_range) / (range_close_ratio.rolling(window=10, min_periods=1).std() + eps)
    
    # Microstructure components
    asymmetric_price_impact = (close - open_) * volume ** 0.8 / ((high - low) ** 0.8 + eps) * ((close - low) / (high - low + eps) - 0.5)
    asymmetric_execution = ((close - open_) ** 2) / ((high - low) ** 2 + eps) * volume ** 0.5 * ((close - low) / (high - low + eps) - 0.5)
    
    asymmetric_microstructure = asymmetric_price_impact * asymmetric_execution
    
    # Core components
    intraday_fractal_core = synchronized_intraday * intraday_cascade
    intraday_persistence_core = intraday_persistence_cascade
    intraday_efficiency_core = intraday_efficiency_ratio
    
    # Base factor
    base_intraday_factor = intraday_fractal_core * intraday_persistence_core * intraday_efficiency_core
    
    # Final alpha calculation
    transmission_alpha_base = base_intraday_factor * intraday_cascade
    enhanced_transmission_alpha = transmission_alpha_base * intraday_efficiency_ratio
    final_alpha = enhanced_transmission_alpha * asymmetric_microstructure
    
    # Handle NaN values
    alpha = final_alpha.fillna(0)
    
    return alpha
