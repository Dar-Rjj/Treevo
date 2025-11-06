import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Helper function for True Range
    def true_range(df):
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))
        return pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # Calculate True Range
    data['tr'] = true_range(data)
    
    # Multi-Scale Price Acceleration
    # Short-term (3-day scale)
    log_ret_1d = np.log(data['close'] / data['close'].shift(1))
    accel_short_price = (log_ret_1d - log_ret_1d.shift(1)) - (log_ret_1d.shift(1) - log_ret_1d.shift(2))
    
    # Medium-term (8-day scale)
    log_ret_4d = np.log(data['close'] / data['close'].shift(4))
    log_ret_8d = np.log(data['close'].shift(4) / data['close'].shift(8))
    log_ret_12d = np.log(data['close'].shift(8) / data['close'].shift(12))
    accel_medium_price = (log_ret_4d - log_ret_8d) - (log_ret_8d - log_ret_12d)
    
    # Long-term (15-day scale)
    log_ret_7d = np.log(data['close'] / data['close'].shift(7))
    log_ret_14d = np.log(data['close'].shift(7) / data['close'].shift(14))
    log_ret_21d = np.log(data['close'].shift(14) / data['close'].shift(21))
    accel_long_price = (log_ret_7d - log_ret_14d) - (log_ret_14d - log_ret_21d)
    
    # Multi-Scale Volume Acceleration
    # Short-term (3-day scale)
    log_vol_1d = np.log(data['volume'] / data['volume'].shift(1))
    accel_short_vol = (log_vol_1d - log_vol_1d.shift(1)) - (log_vol_1d.shift(1) - log_vol_1d.shift(2))
    
    # Medium-term (8-day scale)
    log_vol_4d = np.log(data['volume'] / data['volume'].shift(4))
    log_vol_8d = np.log(data['volume'].shift(4) / data['volume'].shift(8))
    log_vol_12d = np.log(data['volume'].shift(8) / data['volume'].shift(12))
    accel_medium_vol = (log_vol_4d - log_vol_8d) - (log_vol_8d - log_vol_12d)
    
    # Long-term (15-day scale)
    log_vol_7d = np.log(data['volume'] / data['volume'].shift(7))
    log_vol_14d = np.log(data['volume'].shift(7) / data['volume'].shift(14))
    log_vol_21d = np.log(data['volume'].shift(14) / data['volume'].shift(21))
    accel_long_vol = (log_vol_7d - log_vol_14d) - (log_vol_14d - log_vol_21d)
    
    # Acceleration-Memory Integration (15-day lookback)
    def acceleration_persistence(accel_series, window=15):
        persistence = pd.Series(index=accel_series.index, dtype=float)
        for i in range(window, len(accel_series)):
            current_accel = accel_series.iloc[i]
            historical_accels = accel_series.iloc[i-window:i]
            # Calculate similarity and persistence
            similar_patterns = historical_accels[np.abs(historical_accels - current_accel) < np.std(historical_accels)]
            if len(similar_patterns) > 0:
                # Weight by persistence of similar patterns
                persistence.iloc[i] = np.mean(np.sign(similar_patterns) == np.sign(current_accel))
            else:
                persistence.iloc[i] = 0.5  # Neutral persistence
        return persistence
    
    # Calculate persistence for price accelerations
    price_accel_combined = (accel_short_price + accel_medium_price + accel_long_price) / 3
    price_persistence = acceleration_persistence(price_accel_combined)
    
    # Weight current acceleration by historical persistence
    weighted_price_accel = price_accel_combined * price_persistence
    
    # Volatility-Regime Adaptive Processing
    # Multi-Timeframe Volatility State Assessment
    vol_ratio_short = (data['tr'].rolling(window=3).mean() / data['close'])
    vol_ratio_long = (data['tr'].rolling(window=14).mean() / data['close'])
    volatility_ratio = vol_ratio_short / vol_ratio_long
    
    # Regime Classification
    high_vol_regime = volatility_ratio > 1.3
    low_vol_regime = volatility_ratio < 0.7
    
    # Regime-Adaptive Acceleration Processing
    regime_adaptive_accel = pd.Series(index=data.index, dtype=float)
    
    # High Volatility Processing
    high_vol_mask = high_vol_regime & ~vol_ratio_short.isna()
    regime_adaptive_accel[high_vol_mask] = (
        weighted_price_accel[high_vol_mask] * volatility_ratio[high_vol_mask] / vol_ratio_short[high_vol_mask]
    )
    
    # Low Volatility Processing
    low_vol_mask = low_vol_regime & ~vol_ratio_short.isna()
    regime_adaptive_accel[low_vol_mask] = weighted_price_accel[low_vol_mask]
    
    # Normal regime (fallback)
    normal_mask = ~high_vol_regime & ~low_vol_regime & ~vol_ratio_short.isna()
    regime_adaptive_accel[normal_mask] = weighted_price_accel[normal_mask]
    
    # Microstructure Acceleration Enhancement
    # Spread acceleration
    range_ratio = (data['high'] - data['low']) / (data['close'] - data['open']).replace(0, np.nan)
    log_range_ratio = np.log(range_ratio)
    spread_accel = (log_range_ratio - log_range_ratio.shift(1)) - (log_range_ratio.shift(1) - log_range_ratio.shift(2))
    
    # Intraday momentum acceleration
    intraday_ret = data['close'] / data['open'] - 1
    intraday_accel = (intraday_ret - intraday_ret.shift(1)) - (intraday_ret.shift(1) - intraday_ret.shift(2))
    
    # Regime-adaptive microstructure weighting
    microstructure_component = (
        spread_accel.fillna(0) * 0.3 + 
        intraday_accel.fillna(0) * 0.7
    ) * (1 + 0.5 * high_vol_regime.astype(float))
    
    # Price-Memory Context Integration
    # Short-term reversal component
    short_term_low_min = data['low'].rolling(window=3).min()
    short_term_high_max = data['high'].rolling(window=3).max()
    short_term_reversal = (
        (data['close'] - short_term_low_min) / data['close'] - 
        (short_term_high_max - data['close']) / short_term_high_max
    )
    
    # Medium-term reversal component
    medium_term_low_min = data['low'].rolling(window=14).min()
    medium_term_high_max = data['high'].rolling(window=14).max()
    medium_term_reversal = (
        (data['close'] - medium_term_low_min) / data['close'] - 
        (medium_term_high_max - data['close']) / medium_term_high_max
    )
    
    # Historical pattern matching for consistency
    def pattern_consistency_score(reversal_series, window=15):
        consistency = pd.Series(index=reversal_series.index, dtype=float)
        for i in range(window, len(reversal_series)):
            current_reversal = reversal_series.iloc[i]
            historical_reversals = reversal_series.iloc[i-window:i]
            # Calculate consistency of similar reversal patterns
            similar_patterns = historical_reversals[
                np.abs(historical_reversals - current_reversal) < np.std(historical_reversals)
            ]
            if len(similar_patterns) > 0:
                consistency.iloc[i] = np.mean(np.sign(similar_patterns) == np.sign(current_reversal))
            else:
                consistency.iloc[i] = 0.5
        return consistency
    
    short_term_consistency = pattern_consistency_score(short_term_reversal)
    medium_term_consistency = pattern_consistency_score(medium_term_reversal)
    
    # Weight current reversal signals by historical pattern success
    weighted_short_reversal = short_term_reversal * short_term_consistency
    weighted_medium_reversal = medium_term_reversal * medium_term_consistency
    
    # Composite Factor Generation
    # Multi-Scale Acceleration-Memory Score
    acceleration_memory_score = (
        regime_adaptive_accel.fillna(0) * 0.6 + 
        microstructure_component.fillna(0) * 0.4
    )
    
    # Volatility-Adaptive Processing Score
    volatility_adaptive_score = (
        regime_adaptive_accel.fillna(0) * (1 + 0.3 * high_vol_regime.astype(float)) +
        accel_short_vol.fillna(0) * 0.2
    )
    
    # Price-Memory Context Score
    price_memory_score = (
        weighted_short_reversal.fillna(0) * 0.4 + 
        weighted_medium_reversal.fillna(0) * 0.6
    )
    
    # Final Alpha Factor
    final_factor = (
        acceleration_memory_score * 
        volatility_adaptive_score * 
        (1 + price_memory_score)
    )
    
    # Normalize and clean
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan)
    final_factor = (final_factor - final_factor.rolling(window=30, min_periods=10).mean()) / final_factor.rolling(window=30, min_periods=10).std()
    
    return final_factor
