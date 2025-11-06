import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining intraday momentum asymmetry, volume-volatility regime detection,
    price-volume fractal coherence, and range-volume efficiency metrics.
    """
    # Create copy to avoid modifying original dataframe
    data = df.copy()
    
    # Calculate basic price returns and ranges
    data['prev_close'] = data['close'].shift(1)
    data['ret'] = data['close'] / data['prev_close'] - 1
    data['daily_range'] = data['high'] - data['low']
    data['intraday_move'] = data['close'] - data['open']
    
    # 1. Intraday Momentum Asymmetry
    data['gap_capture'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['early_strength'] = (data['high'] - data['open']) / data['open']
    data['midday_fade'] = (data['close'] - data['high']) / data['high']
    
    # Morning persistence (simplified using daily data)
    data['morning_ret_sign'] = np.sign(data['intraday_move'])
    data['morning_persistence'] = data['morning_ret_sign'].rolling(window=3, min_periods=1).apply(
        lambda x: np.sum(x == x.iloc[-1]) if len(x) > 0 else 0, raw=False
    )
    
    # Afternoon reversal frequency
    data['afternoon_reversal'] = (data['close'] < data['open']).rolling(window=5, min_periods=1).sum()
    
    # Asymmetry signal
    data['morning_strength'] = data['gap_capture'] + data['early_strength'] + data['morning_persistence'] / 3
    data['afternoon_weakness'] = data['midday_fade'] + (data['afternoon_reversal'] / 5)
    data['momentum_asymmetry'] = data['morning_strength'] - data['afternoon_weakness']
    
    # 2. Volume-Volatility Regime Switch
    data['volume_persistence'] = data['volume'] / data['volume'].shift(1)
    data['range_efficiency'] = data['intraday_move'] / data['daily_range']
    data['gap_persistence'] = abs(data['open'] - data['prev_close']) / data['daily_range']
    
    # Volatility clustering
    data['vol_3d'] = data['daily_range'].rolling(window=3, min_periods=1).std()
    data['vol_10d'] = data['daily_range'].rolling(window=10, min_periods=1).std()
    data['volatility_clustering'] = data['vol_3d'] / data['vol_10d']
    
    # Regime detection
    data['volume_ma'] = data['volume'].rolling(window=10, min_periods=1).mean()
    data['volatility_ma'] = data['daily_range'].rolling(window=10, min_periods=1).mean()
    data['high_volume'] = data['volume'] > data['volume_ma']
    data['low_volatility'] = data['daily_range'] < data['volatility_ma']
    data['regime_score'] = (data['high_volume'].astype(int) + data['low_volatility'].astype(int)) * data['range_efficiency']
    
    # 3. Price-Volume Fractal Coherence
    data['micro_fractal'] = data['daily_range'] / abs(data['intraday_move']).replace(0, 1e-6)
    data['meso_fractal'] = data['close'].rolling(window=3, min_periods=1).std() / data['close'].rolling(window=10, min_periods=1).std()
    
    # Volume fractal dimension (simplified)
    data['volume_fractal'] = np.log(data['volume'] / data['volume'].shift(1).replace(0, 1e-6))
    
    # Price-volume scaling correlation
    data['price_change'] = data['close'].pct_change()
    data['volume_change'] = data['volume'].pct_change()
    data['price_volume_corr'] = data['price_change'].rolling(window=8, min_periods=1).corr(data['volume_change'])
    
    # Coherence signal
    data['fractal_coherence'] = (data['micro_fractal'] + data['meso_fractal'] + data['price_volume_corr']) / 3
    
    # 4. Range-Volume Efficiency Frontier
    data['price_efficiency'] = abs(data['intraday_move']) / data['daily_range'].replace(0, 1e-6)
    data['volume_efficiency'] = data['volume'] / data['daily_range'].replace(0, 1e-6)
    
    # Efficiency ratio and momentum
    data['efficiency_ratio'] = data['price_efficiency'] / data['volume_efficiency'].replace(0, 1e-6)
    data['efficiency_ma'] = data['efficiency_ratio'].rolling(window=10, min_periods=1).mean()
    data['frontier_distance'] = data['efficiency_ratio'] - data['efficiency_ma']
    data['efficiency_momentum'] = data['efficiency_ratio'] / data['efficiency_ratio'].shift(1).replace(0, 1e-6)
    
    # Combine all components into final alpha factor
    alpha = (
        0.3 * data['momentum_asymmetry'] +
        0.25 * data['regime_score'] +
        0.25 * data['fractal_coherence'] +
        0.2 * data['frontier_distance']
    )
    
    # Normalize and clean
    alpha = (alpha - alpha.rolling(window=20, min_periods=1).mean()) / alpha.rolling(window=20, min_periods=1).std()
    alpha = alpha.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
    
    return alpha
