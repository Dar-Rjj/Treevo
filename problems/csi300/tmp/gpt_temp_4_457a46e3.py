import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility-Regime Detection
    data['daily_range_vol'] = (data['high'] - data['low']) / data['close']
    data['microstructure_depth'] = data['amount'] / (data['high'] - data['low'])
    
    # Calculate rolling percentiles for regime classification
    data['vol_percentile'] = data['daily_range_vol'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['depth_percentile'] = data['microstructure_depth'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Regime classification: 1 for high volatility, 0 for low volatility
    data['vol_regime'] = ((data['vol_percentile'] > 0.75) & (data['depth_percentile'] < 0.25)).astype(int)
    
    # Efficiency-Adjusted Intraday Reversal
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    data['volume_efficiency'] = data['amount'] / data['volume']
    data['efficiency_reversal'] = data['intraday_return'] * data['volume_efficiency']
    
    # Volume-Price Convergence Analysis
    data['price_acceleration'] = (data['close'] - 2 * data['close'].shift(1) + data['close'].shift(2)) / (data['high'] - data['low'])
    
    # Volume momentum using rolling median
    data['volume_median_9d'] = data['volume'].rolling(window=9, min_periods=5).median().shift(1)
    data['volume_momentum'] = data['volume'] / data['volume_median_9d']
    
    data['convergence_signal'] = data['price_acceleration'] * data['volume_momentum']
    
    # Medium-term convergence signal (3-day window)
    data['convergence_3d'] = data['convergence_signal'].rolling(window=3, min_periods=2).mean()
    
    # Multi-Timeframe Signal Integration with regime-dependent weighting
    # High volatility regime: emphasize reversal (weight = 0.7)
    # Low volatility regime: emphasize convergence (weight = 0.3)
    data['reversal_weight'] = np.where(data['vol_regime'] == 1, 0.7, 0.3)
    data['convergence_weight'] = 1 - data['reversal_weight']
    
    # Final Factor Construction
    data['final_factor'] = (
        data['reversal_weight'] * data['efficiency_reversal'] + 
        data['convergence_weight'] * data['convergence_3d']
    )
    
    # Return the factor series
    return data['final_factor']
