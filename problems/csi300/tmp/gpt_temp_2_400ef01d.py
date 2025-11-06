import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Acceleration Analysis
    data['short_term_acceleration'] = (data['close'] / data['close'].shift(3) - 1) - (data['close'].shift(3) / data['close'].shift(6) - 1)
    data['medium_term_acceleration'] = (data['close'] / data['close'].shift(5) - 1) - (data['close'].shift(5) / data['close'].shift(10) - 1)
    data['acceleration_divergence'] = data['short_term_acceleration'] - data['medium_term_acceleration']
    
    # Volatility Regime Detection
    # True Range calculation
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['prev_close']),
            np.abs(data['low'] - data['prev_close'])
        )
    )
    
    # Average True Range (20-day)
    data['ATR'] = data['true_range'].rolling(window=20, min_periods=20).mean()
    
    # Volatility regime classification
    def get_volatility_regime(row):
        if pd.isna(row['ATR']) or pd.isna(row['true_range']):
            return 'normal'
        
        recent_tr = data['true_range'].shift(1).rolling(window=20, min_periods=20).apply(
            lambda x: np.percentile(x.dropna(), 80) if len(x.dropna()) >= 20 else np.nan, 
            raw=False
        ).loc[row.name]
        
        if pd.isna(recent_tr):
            return 'normal'
        
        if row['true_range'] > recent_tr:
            return 'high'
        elif row['true_range'] < np.percentile(data['true_range'].shift(1).dropna().tail(20), 20):
            return 'low'
        else:
            return 'normal'
    
    data['volatility_regime'] = data.apply(get_volatility_regime, axis=1)
    data['volatility_clustering'] = data['true_range'] / data['true_range'].shift(1)
    
    # Microstructure Noise Assessment
    data['bid_ask_spread_proxy'] = 2 * np.abs(data['close'] - (data['high'] + data['low']) / 2) / (data['high'] - data['low'])
    data['price_discreteness'] = (np.mod(data['close'] * 100, 1)) / data['close']
    data['overnight_gap_efficiency'] = np.abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['opening_auction_deviation'] = np.abs(data['open'] - (data['high'] + data['low']) / 2) / (data['high'] - data['low'])
    
    # Volume-Volatility Co-Movement Analysis
    data['volume_momentum'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_acceleration'] = (data['volume'] / data['volume'].shift(5) - 1) - (data['volume'].shift(5) / data['volume'].shift(10) - 1)
    
    # Volume-volatility alignment
    def get_volume_volatility_alignment(row):
        if pd.isna(row['true_range']) or pd.isna(row['ATR']) or pd.isna(row['volume']) or pd.isna(row['volume'].shift(5)):
            return 'neutral'
        
        if row['true_range'] > row['ATR'] and row['volume'] > data['volume'].shift(5).loc[row.name]:
            return 'high_high'
        elif row['true_range'] < row['ATR'] and row['volume'] < data['volume'].shift(5).loc[row.name]:
            return 'low_low'
        else:
            return 'divergence'
    
    data['volume_volatility_alignment'] = data.apply(get_volume_volatility_alignment, axis=1)
    
    # Volume regime persistence
    def calculate_volume_persistence(row):
        if pd.isna(row.name):
            return np.nan
        
        recent_volumes = []
        for i in range(5):
            if row.name - pd.Timedelta(days=i) in data.index:
                vol_idx = data.index.get_loc(row.name) - i
                if vol_idx >= 0 and vol_idx - 1 >= 0:
                    recent_volumes.append(data.iloc[vol_idx]['volume'] > data.iloc[vol_idx - 1]['volume'])
        
        return sum(recent_volumes) if recent_volumes else np.nan
    
    data['volume_persistence'] = data.index.map(lambda x: calculate_volume_persistence(data.loc[x]) if x in data.index else np.nan)
    
    # Regime-Adaptive Signal Construction
    def get_base_signal(row):
        if pd.isna(row['acceleration_divergence']):
            return np.nan
        
        if row['volatility_regime'] == 'high':
            return row['acceleration_divergence'] * (row['volatility_clustering'] if not pd.isna(row['volatility_clustering']) else 1)
        elif row['volatility_regime'] == 'low':
            return row['acceleration_divergence'] / (1 + (row['price_discreteness'] if not pd.isna(row['price_discreteness']) else 0))
        else:  # normal
            return row['acceleration_divergence'] * (row['volume_momentum'] if not pd.isna(row['volume_momentum']) else 1)
    
    data['base_signal'] = data.apply(get_base_signal, axis=1)
    
    # Microstructure filtering
    data['spread_adjusted_signal'] = data['base_signal'] / (1 + data['bid_ask_spread_proxy'].fillna(0))
    data['gap_efficiency_weighted'] = data['spread_adjusted_signal'] * (1 - data['overnight_gap_efficiency'].fillna(0))
    data['microstructure_filtered'] = data['gap_efficiency_weighted'] * (1 - data['opening_auction_deviation'].fillna(0))
    
    # Volume-volatility regime enhancement
    def apply_volume_enhancement(row):
        if pd.isna(row['microstructure_filtered']):
            return np.nan
        
        signal = row['microstructure_filtered']
        
        if row['volume_volatility_alignment'] == 'high_high' and not pd.isna(row['volume_acceleration']):
            signal *= row['volume_acceleration']
        elif row['volume_volatility_alignment'] == 'divergence' and not pd.isna(row['volume_persistence']):
            signal *= -row['volume_persistence']
        
        # Regime transition detection
        if not pd.isna(row['volatility_clustering']):
            signal *= row['volatility_clustering']
            
        return signal
    
    data['regime_enhanced_signal'] = data.apply(apply_volume_enhancement, axis=1)
    
    # Combined Alpha Factor
    data['base_acceleration_condition'] = data['acceleration_divergence'] > 0
    
    # Volume confirmation multiplier
    def get_volume_multiplier(row):
        if pd.isna(row['volume_acceleration']) or pd.isna(row['volume_persistence']):
            return 1
        
        if row['volume_acceleration'] > 0 and row['volume_volatility_alignment'] == 'high_high':
            return 1.5  # Strong confirmation
        elif row['volume_persistence'] > 3:
            return 1.2  # Weak confirmation
        elif row['volume_volatility_alignment'] == 'divergence':
            return 0.8  # Negative confirmation
        else:
            return 1.0
    
    data['volume_multiplier'] = data.apply(get_volume_multiplier, axis=1)
    
    # Final factor
    data['final_factor'] = (
        data['base_acceleration_condition'].astype(float) * 
        data['regime_enhanced_signal'].fillna(0) * 
        data['volume_multiplier']
    )
    
    return data['final_factor']
