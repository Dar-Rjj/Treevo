import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Hierarchical Volatility Compression with Asymmetric Order Flow alpha factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(60, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # Multi-Frequency Volatility Compression Analysis
        # Short-Term Compression (1-5 days)
        morning_range = current_data['high'] - current_data['low']
        morning_efficiency = (current_data['close'] - current_data['open']) / np.where(morning_range == 0, 1, morning_range)
        
        # 5-day rolling calculations
        morning_range_5d_avg = morning_range.rolling(window=5, min_periods=3).mean()
        compression_intensity = morning_range / np.where(morning_range_5d_avg == 0, 1, morning_range_5d_avg)
        
        # Volatility persistence (correlation between morning and afternoon movements)
        morning_move = current_data['close'] - current_data['open']
        afternoon_move = current_data['close'] - (current_data['high'] + current_data['low']) / 2
        volatility_persistence = morning_move.rolling(window=5, min_periods=3).corr(afternoon_move)
        
        # Medium-Term Compression (5-20 days)
        daily_range = current_data['high'] - current_data['low']
        range_5d_avg = daily_range.rolling(window=5, min_periods=3).mean()
        multi_day_compression = daily_range / np.where(range_5d_avg == 0, 1, range_5d_avg)
        
        # Volatility regime consistency (compression persistence)
        volatility_regime = compression_intensity.rolling(window=5, min_periods=3).std()
        
        # Long-Term Compression Context (20-60 days)
        vol_20d = daily_range.rolling(window=20, min_periods=10).std()
        vol_60d = daily_range.rolling(window=60, min_periods=30).std()
        extended_compression = vol_20d / np.where(vol_60d == 0, 1, vol_60d)
        
        # Historical compression benchmark
        hist_compression = daily_range / daily_range.rolling(window=60, min_periods=30).mean()
        
        # Asymmetric Order Flow Framework
        # Directional Volume Pressure
        returns = current_data['close'].pct_change()
        up_days = returns > 0
        down_days = returns < 0
        
        up_volume_concentration = np.where(
            current_data['volume'].rolling(window=10, min_periods=5).sum() == 0, 0,
            current_data['volume'][up_days].rolling(window=10, min_periods=5).sum() / 
            current_data['volume'].rolling(window=10, min_periods=5).sum()
        )
        
        avg_volume = current_data['volume'].rolling(window=10, min_periods=5).mean()
        down_volume_intensity = np.where(
            avg_volume == 0, 0,
            current_data['volume'][down_days] / avg_volume
        )
        
        # Opening pressure asymmetry
        typical_range = (current_data['high'] - current_data['low']).rolling(window=10, min_periods=5).mean()
        opening_pressure = (current_data['open'] - current_data['close'].shift(1)) / np.where(typical_range == 0, 1, typical_range)
        
        # Volume-Weighted Imbalance
        vwap = (current_data['high'] + current_data['low'] + current_data['close']) / 3
        price_distance = np.abs(current_data['close'] - vwap)
        volume_weighted_distance = (current_data['volume'] * price_distance) / np.where(current_data['volume'] == 0, 1, current_data['volume'])
        
        # Order flow concentration
        price_moves = np.abs(current_data['close'].pct_change())
        top_10_threshold = price_moves.rolling(window=20, min_periods=10).quantile(0.9)
        high_move_days = price_moves > top_10_threshold
        order_flow_concentration = np.where(
            current_data['volume'].rolling(window=10, min_periods=5).sum() == 0, 0,
            current_data['volume'][high_move_days].rolling(window=10, min_periods=5).sum() / 
            current_data['volume'].rolling(window=10, min_periods=5).sum()
        )
        
        # Multi-period imbalance (3-day window)
        volume_imbalance = (current_data['volume'][up_days].rolling(window=3, min_periods=2).sum() - 
                           current_data['volume'][down_days].rolling(window=3, min_periods=2).sum()) / \
                          current_data['volume'].rolling(window=3, min_periods=2).sum()
        
        # Cross-Frequency Efficiency Metrics
        # Multi-timeframe price discovery
        overnight_vol = np.abs(current_data['open'] - current_data['close'].shift(1)).rolling(window=10, min_periods=5).std()
        opening_efficiency = np.abs(current_data['open'] - current_data['close'].shift(1)) / np.where(overnight_vol == 0, 1, overnight_vol)
        
        intraday_vol = (current_data['high'] - current_data['low']).rolling(window=10, min_periods=5).std()
        intraday_efficiency = np.abs(current_data['close'] - vwap) / np.where(intraday_vol == 0, 1, intraday_vol)
        
        # Volatility-Volume co-movement
        vol_change = daily_range.pct_change()
        volume_change = current_data['volume'].pct_change()
        co_movement = vol_change.rolling(window=10, min_periods=5).corr(volume_change)
        
        # Regime-Adaptive Signal Integration
        # Volatility compression regime processing
        compression_weighting = volume_weighted_distance * extended_compression
        efficiency_multiplier = (opening_efficiency + intraday_efficiency) / 2
        
        # Volume asymmetry regime processing
        directional_weighting = up_volume_concentration * volume_imbalance
        imbalance_persistence = volume_imbalance.rolling(window=5, min_periods=3).std()
        
        # Cross-frequency confirmation
        short_term_signal = compression_intensity * volume_imbalance
        medium_term_context = multi_day_compression * imbalance_persistence
        long_term_validation = extended_compression * hist_compression
        
        # Hierarchical Alpha Synthesis
        # Core compression-asymmetry factor
        primary_signal = extended_compression * volume_imbalance
        secondary_enhancement = efficiency_multiplier * compression_weighting
        
        # Robustness layer
        multi_timeframe_agreement = (short_term_signal.rolling(window=5, min_periods=3).mean() + 
                                   medium_term_context.rolling(window=10, min_periods=5).mean() + 
                                   long_term_validation.rolling(window=20, min_periods=10).mean()) / 3
        
        # Final alpha calculation
        hierarchical_compression = (extended_compression * 0.4 + multi_day_compression * 0.3 + compression_intensity * 0.3)
        asymmetric_flow = (volume_imbalance * 0.4 + directional_weighting * 0.3 + order_flow_concentration * 0.3)
        
        integrated_alpha = hierarchical_compression * asymmetric_flow * multi_timeframe_agreement
        
        # Assign current value
        if not pd.isna(integrated_alpha.iloc[-1]):
            alpha.iloc[i] = integrated_alpha.iloc[-1]
    
    return alpha
