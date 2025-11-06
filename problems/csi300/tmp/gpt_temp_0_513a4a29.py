import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate required rolling statistics
    data['volume_avg_3'] = data['volume'].shift(1).rolling(window=3, min_periods=1).mean()
    data['volume_avg_5'] = data['volume'].shift(1).rolling(window=5, min_periods=1).mean()
    data['volume_avg_10'] = data['volume'].shift(1).rolling(window=10, min_periods=1).mean()
    
    # Price returns
    data['ret'] = data['close'] / data['close'].shift(1) - 1
    data['volume_ret'] = data['volume'] / data['volume'].shift(1) - 1
    
    # Volatility measures
    data['vol_3d'] = data['ret'].rolling(window=3, min_periods=1).std()
    data['vol_5d'] = data['ret'].rolling(window=5, min_periods=1).std()
    data['vol_10d'] = data['ret'].rolling(window=10, min_periods=1).std()
    data['vol_20d'] = data['ret'].rolling(window=20, min_periods=1).std()
    
    # Range calculations
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['prev_range'] = (data['high'].shift(1) - data['low'].shift(1)) / data['close'].shift(1)
    
    # VWAP calculation
    data['vwap'] = data['amount'] / data['volume']
    
    # Calculate individual components
    for i in range(len(data)):
        if i < 5:  # Skip first few days for stability
            result.iloc[i] = 0
            continue
            
        current = data.iloc[i]
        
        # Intraday Microstructure Patterns
        opening_vol_vol = current['daily_range'] * current['volume'] / max(data['volume_avg_3'].iloc[i], 1)
        midday_compression = (current['daily_range'] / max(data['prev_range'].iloc[i], 0.001)) * (current['volume'] / max(data['volume'].iloc[i-1], 1))
        closing_expansion = ((current['close'] - current['open']) / max(current['high'] - current['low'], 0.001)) * (current['volume'] / max(data['volume_avg_5'].iloc[i], 1))
        
        # Trade Flow Impact Analysis
        volatility_clustering = current['vol_3d'] * (current['volume'] / max(current['amount'], 1))
        
        # Price impact correlation (5-day window)
        if i >= 10:
            ret_window = data['ret'].iloc[i-4:i+1]
            vol_ret_window = data['volume_ret'].iloc[i-4:i+1]
            price_impact_corr = np.corrcoef(ret_window, vol_ret_window)[0,1] if len(ret_window) > 1 and not np.isnan(ret_window).any() and not np.isnan(vol_ret_window).any() else 0
        else:
            price_impact_corr = 0
            
        price_impact_asymmetry = current['daily_range'] * price_impact_corr
        
        # Trade direction persistence (5-day)
        if i >= 10:
            sign_matches = 0
            for j in range(5):
                idx = i - j
                price_sign = np.sign(data['close'].iloc[idx] - data['close'].iloc[idx-1])
                volume_sign = np.sign(data['volume'].iloc[idx] - data['volume'].iloc[idx-1])
                if price_sign == volume_sign and price_sign != 0:
                    sign_matches += 1
            trade_direction_persistence = sign_matches / 5
        else:
            trade_direction_persistence = 0
            
        # Volume-Weighted Trend Signals
        directional_movement = ((current['high'] - data['high'].iloc[i-1]) - (data['low'].iloc[i-1] - current['low'])) * (current['volume'] / max(data['volume'].iloc[i-1], 1))
        vwap_deviation = ((current['vwap'] - (current['high'] + current['low'])/2) / max(current['close'], 0.001)) * np.sign(current['close'] - data['close'].iloc[i-1])
        
        # Multi-Timeframe Volume Structure
        short_term_vol = current['vol_3d'] * (current['volume'] / max(data['volume_avg_3'].iloc[i], 1))
        
        # Volume regime (10-day count)
        if i >= 30:
            regime_count = 0
            for j in range(10):
                idx = i - j
                if data['vol_5d'].iloc[idx] > data['vol_20d'].iloc[idx]:
                    regime_count += 1
            volume_regime = (regime_count / 10) * (current['volume'] / max(data['volume_avg_10'].iloc[i], 1))
        else:
            volume_regime = 0
            
        volatility_momentum = ((current['vol_5d'] - current['vol_10d']) / max(current['vol_5d'], 0.001)) * (current['volume'] / max(data['volume'].iloc[i-1], 1))
        
        # Regime-Adaptive Processing
        if i >= 10:
            ret_corr_window1 = data['ret'].iloc[i-4:i+1]
            ret_corr_window2 = data['ret'].iloc[i-5:i]
            high_vol_reversal_corr = -np.corrcoef(ret_corr_window1, ret_corr_window2)[0,1] if len(ret_corr_window1) > 1 and len(ret_corr_window2) > 1 else 0
        else:
            high_vol_reversal_corr = 0
            
        high_vol_reversal = high_vol_reversal_corr * (current['volume'] / max(data['volume'].iloc[i-1], 1))
        
        large_trade_momentum = ((current['close'] / data['close'].iloc[i-5] - 1) / max(current['vol_5d'], 0.001)) * (current['volume'] / max(current['amount'], 1))
        
        # Quote breakout
        prev_high_max = max(data['high'].iloc[i-5:i])
        quote_breakout = ((current['high'] - prev_high_max) / max(current['close'], 0.001)) * (current['volume'] / max(data['volume_avg_5'].iloc[i], 1))
        
        # Overnight Gap Integration
        volume_weighted_gap = ((current['open'] / data['close'].iloc[i-1] - 1)) * (current['volume'] / max(data['volume'].iloc[i-1], 1))
        gap_microstructure = np.sign(current['open'] - data['close'].iloc[i-1]) * np.sign(current['close'] - current['open']) * (current['volume'] / max(data['volume'].iloc[i-1], 1))
        opening_persistence = ((current['open'] / data['close'].iloc[i-1] - 1)) * (current['volume'] / max(data['volume_avg_3'].iloc[i], 1)) * current['daily_range']
        
        # Multi-Frequency Integration
        intraday_microstructure = ((current['close'] - current['open']) / max(current['high'] - current['low'], 0.001)) * (current['volume'] / max(data['volume'].iloc[i-1], 1))
        
        # Medium-term persistence (10-day)
        if i >= 20:
            sign_matches_10d = 0
            for j in range(10):
                idx = i - j
                price_sign = np.sign(data['close'].iloc[idx] - data['close'].iloc[idx-1])
                volume_sign = np.sign(data['volume'].iloc[idx] - data['volume'].iloc[idx-1])
                if price_sign == volume_sign and price_sign != 0:
                    sign_matches_10d += 1
            medium_term_persistence = sign_matches_10d / 10
        else:
            medium_term_persistence = 0
            
        # Long-term efficiency (20-day correlation)
        if i >= 25:
            ret_window_20d = data['ret'].iloc[i-19:i+1]
            vol_ret_window_20d = data['volume_ret'].iloc[i-19:i+1]
            long_term_efficiency = np.corrcoef(ret_window_20d, vol_ret_window_20d)[0,1] if len(ret_window_20d) > 1 and not np.isnan(ret_window_20d).any() and not np.isnan(vol_ret_window_20d).any() else 0
        else:
            long_term_efficiency = 0
            
        # Combine all components with equal weights
        components = [
            opening_vol_vol, midday_compression, closing_expansion,
            volatility_clustering, price_impact_asymmetry, trade_direction_persistence,
            directional_movement, vwap_deviation, trade_direction_persistence,
            short_term_vol, volume_regime, volatility_momentum,
            high_vol_reversal, large_trade_momentum, quote_breakout,
            volume_weighted_gap, gap_microstructure, opening_persistence,
            intraday_microstructure, medium_term_persistence, long_term_efficiency
        ]
        
        # Remove NaN values and calculate final factor
        valid_components = [c for c in components if not np.isnan(c) and not np.isinf(c)]
        if valid_components:
            result.iloc[i] = np.mean(valid_components)
        else:
            result.iloc[i] = 0
    
    # Fill initial NaN values
    result = result.fillna(0)
    
    return result
