import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor based on volatility-volume interaction, intraday momentum patterns,
    exhaustion reversal detection, efficiency-based adaptation, and pressure breakout signals.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate returns
    data['returns'] = data['close'].pct_change()
    
    # Volatility regime classification
    vol_window = 20
    data['volatility'] = data['returns'].rolling(window=vol_window, min_periods=10).std()
    data['vol_regime'] = data['volatility'] > data['volatility'].rolling(window=60, min_periods=30).mean()
    
    # Volume behavior analysis
    data['volume_ma'] = data['volume'].rolling(window=10, min_periods=5).mean()
    data['volume_trend'] = data['volume'] / data['volume_ma'] - 1
    data['volume_clustering'] = (data['volume'] > data['volume'].rolling(window=20, min_periods=10).quantile(0.7)).astype(int)
    
    # Intraday momentum patterns
    data['intraday_range'] = (data['high'] - data['low']) / data['open']
    data['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Morning session proxy (first 30% of trading range)
    data['morning_strength'] = (data['open'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Afternoon session proxy (last 30% of trading range)
    data['afternoon_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Exhaustion reversal detection
    data['volume_spike'] = data['volume'] / data['volume'].rolling(window=10, min_periods=5).mean() - 1
    data['price_extension'] = (data['close'] - data['close'].rolling(window=10, min_periods=5).mean()) / data['close'].rolling(window=10, min_periods=5).std()
    
    # Efficiency-based adaptation
    data['intraday_efficiency'] = (data['close'] - data['open']) / data['intraday_range'].replace(0, np.nan)
    data['trend_strength'] = data['close'].rolling(window=10, min_periods=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] / np.std(x) if len(x) >= 5 else np.nan
    )
    
    # Pressure breakout signals
    data['price_pressure'] = (data['close'] - data['open']) * data['volume']
    data['cumulative_pressure'] = data['price_pressure'].rolling(window=5, min_periods=3).sum()
    data['pressure_breakout'] = (data['cumulative_pressure'] > data['cumulative_pressure'].rolling(window=20, min_periods=10).quantile(0.8)).astype(int)
    
    # Signal construction components
    # Volatility-volume interaction
    data['vol_volume_interaction'] = data['volume_trend'] * data['volatility']
    
    # Intraday momentum integration
    data['cross_session_alignment'] = data['morning_strength'] * data['afternoon_momentum']
    
    # Exhaustion signals
    data['exhaustion_signal'] = -data['volume_spike'] * data['price_extension']
    
    # Efficiency adaptation
    data['efficiency_weight'] = 1 / (1 + np.abs(data['intraday_efficiency']))
    
    # Pressure breakout refinement
    data['pressure_signal'] = data['cumulative_pressure'] * data['pressure_breakout']
    
    # Final factor construction with regime adaptation
    data['factor'] = (
        data['vol_volume_interaction'].fillna(0) * 
        (1 + data['vol_regime'].fillna(0)) +  # Higher weight in high vol regimes
        data['cross_session_alignment'].fillna(0) * 
        data['efficiency_weight'].fillna(1) +  # Efficiency-adjusted momentum
        data['exhaustion_signal'].fillna(0) * 
        (1 + np.abs(data['trend_strength'].fillna(0))) +  # Stronger exhaustion in trends
        data['pressure_signal'].fillna(0)
    )
    
    # Normalize the factor
    factor_series = data['factor'].copy()
    factor_series = (factor_series - factor_series.rolling(window=60, min_periods=30).mean()) / factor_series.rolling(window=60, min_periods=30).std()
    
    return factor_series
