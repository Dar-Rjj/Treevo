import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Volatility-Adjusted Momentum with Volume Confirmation alpha factor
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Multi-Timeframe Price Momentum
    # Short-term (3-day)
    data['price_return_3d'] = data['close'] / data['close'].shift(3) - 1
    data['price_acceleration'] = (data['close'] / data['close'].shift(1)) - (data['close'].shift(1) / data['close'].shift(2))
    
    # Medium-term (8-day)
    data['price_return_8d'] = data['close'] / data['close'].shift(8) - 1
    
    # Price Consistency (count days with same direction as 8d return)
    def calculate_price_consistency(row):
        if pd.isna(row['price_return_8d']):
            return np.nan
        sign_8d = np.sign(row['price_return_8d'])
        count = 0
        for i in range(8):
            if i > 0:
                daily_return = data.loc[row.name, 'close'] / data.loc[data.index[data.index.get_loc(row.name)-i-1], 'close'] - 1
                if np.sign(daily_return) == sign_8d:
                    count += 1
        return count
    
    data['price_consistency'] = data.apply(calculate_price_consistency, axis=1)
    
    # Multi-Timeframe Alignment
    data['direction_agreement'] = (np.sign(data['price_return_3d']) == np.sign(data['price_return_8d'])).astype(int)
    data['acceleration_confirmation'] = (np.sign(data['price_acceleration']) == np.sign(data['price_return_8d'])).astype(int)
    data['aligned_momentum'] = data['direction_agreement'] * data['acceleration_confirmation'] * (data['price_return_3d'] + data['price_return_8d'])
    
    # Volatility-Adjusted Components
    # True Range
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['close'].shift(1)),
            np.abs(data['low'] - data['close'].shift(1))
        )
    )
    
    # 5-day Average True Range
    data['atr_5d'] = data['true_range'].rolling(window=5).mean()
    
    # 10-day Volatility Ratio
    def calculate_volatility_ratio(idx):
        if idx < 10:
            return np.nan
        recent_vol = data['close'].iloc[idx-4:idx+1].pct_change().std()
        previous_vol = data['close'].iloc[idx-9:idx-4].pct_change().std()
        if previous_vol == 0:
            return np.nan
        return recent_vol / previous_vol
    
    data['volatility_ratio_10d'] = [calculate_volatility_ratio(i) for i in range(len(data))]
    
    # Volatility-Scaled Factors
    data['volatility_scaled_short_term'] = data['price_return_3d'] / (data['atr_5d'] / data['close'].shift(1) + 0.001)
    data['volatility_scaled_medium_term'] = data['price_return_8d'] / (data['atr_5d'] / data['close'].shift(1) + 0.001)
    data['volatility_scaled_aligned'] = data['aligned_momentum'] / (data['atr_5d'] / data['close'].shift(1) + 0.001)
    data['volatility_regime'] = (data['volatility_ratio_10d'] > 1.0).astype(float)
    
    # Volume Confirmation System
    # Multi-Timeframe Volume Momentum
    data['volume_short_term'] = data['volume'] / data['volume'].shift(3) - 1
    data['volume_medium_term'] = data['volume'] / data['volume'].shift(8) - 1
    data['volume_alignment'] = (np.sign(data['volume_short_term']) == np.sign(data['volume_medium_term'])).astype(int)
    
    # Volume Persistence
    def calculate_volume_trend_3d(row):
        if pd.isna(row['volume']) or pd.isna(data.loc[data.index[data.index.get_loc(row.name)-3], 'volume']):
            return np.nan
        count = 0
        for i in range(3):
            if i > 0:
                if data.loc[row.name, 'volume'] > data.loc[data.index[data.index.get_loc(row.name)-i], 'volume']:
                    count += 1
        return count
    
    def calculate_volume_trend_8d(row):
        if pd.isna(row['volume']) or pd.isna(data.loc[data.index[data.index.get_loc(row.name)-8], 'volume']):
            return np.nan
        count = 0
        for i in range(8):
            if i > 0:
                if data.loc[row.name, 'volume'] > data.loc[data.index[data.index.get_loc(row.name)-i], 'volume']:
                    count += 1
        return count
    
    data['volume_trend_3d'] = data.apply(calculate_volume_trend_3d, axis=1)
    data['volume_trend_8d'] = data.apply(calculate_volume_trend_8d, axis=1)
    data['volume_consistency'] = data['volume_trend_3d'] * data['volume_trend_8d']
    
    # Volume-Price Confirmation
    data['short_term_confirmation'] = data['price_return_3d'] * np.sign(data['volume_short_term']) * data['volume_alignment']
    data['medium_term_confirmation'] = data['price_return_8d'] * np.sign(data['volume_medium_term']) * data['volume_consistency']
    data['multi_timeframe_confirmation'] = data['short_term_confirmation'] * data['medium_term_confirmation']
    
    # Amount Flow Persistence
    # Directional Flow Analysis
    data['daily_flow'] = data['amount'] * np.sign(data['close'] - data['close'].shift(1))
    data['flow_sum_3d'] = data['daily_flow'].rolling(window=3).sum()
    data['flow_price_alignment'] = (np.sign(data['flow_sum_3d']) == np.sign(data['price_return_3d'])).astype(int)
    
    # Flow Persistence Metrics
    def calculate_flow_consistency(row):
        if pd.isna(row['daily_flow']):
            return np.nan
        count = 0
        for i in range(3):
            if i > 0:
                current_sign = np.sign(data.loc[row.name, 'daily_flow'])
                prev_sign = np.sign(data.loc[data.index[data.index.get_loc(row.name)-i], 'daily_flow'])
                if current_sign == prev_sign:
                    count += 1
        return count
    
    data['flow_consistency'] = data.apply(calculate_flow_consistency, axis=1)
    data['flow_magnitude_persistence'] = data['amount'].rolling(window=3).sum() / data['amount'].shift(3).rolling(window=3).sum()
    data['persistent_flow'] = data['flow_consistency'] * data['flow_magnitude_persistence'] * data['flow_sum_3d']
    
    # Composite Alpha Factors
    # Core Volatility-Adjusted Momentum
    data['core_momentum'] = (data['volatility_scaled_short_term'] + data['volatility_scaled_medium_term']) / 2
    data['volume_confirmed_core'] = data['core_momentum'] * data['multi_timeframe_confirmation']
    data['final_core'] = data['volume_confirmed_core'] * (1 + 0.5 * data['volatility_regime'])
    
    # Persistent Volume-Price Factor
    data['volume_price_alignment'] = data['short_term_confirmation'] * data['medium_term_confirmation']
    data['volatility_adjusted_vp'] = data['volume_price_alignment'] / (data['atr_5d'] / data['close'].shift(1) + 0.001)
    data['flow_enhanced_vp'] = data['volatility_adjusted_vp'] * np.sign(data['persistent_flow'])
    
    # Multi-Timeframe Convergence Factor
    data['price_convergence'] = data['direction_agreement'] * data['acceleration_confirmation'] * data['price_consistency']
    data['volume_convergence'] = data['volume_alignment'] * data['volume_consistency']
    data['final_convergence'] = data['price_convergence'] * data['volume_convergence'] * data['flow_price_alignment']
    
    # Final composite alpha factor (weighted combination)
    alpha = (0.4 * data['final_core'] + 
             0.35 * data['flow_enhanced_vp'] + 
             0.25 * data['final_convergence'])
    
    return alpha
