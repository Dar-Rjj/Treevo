import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Regime-Adaptive Price-Volume Momentum Divergence factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Detection
    # Calculate daily true range
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['prev_close']),
            np.abs(data['low'] - data['prev_close'])
        )
    )
    data['volatility_ratio'] = data['true_range'] / data['prev_close']
    
    # Rolling volatility baseline (20-day median)
    data['volatility_baseline'] = data['volatility_ratio'].rolling(window=20, min_periods=10).median()
    
    # Volatility regime classification
    conditions_vol = [
        data['volatility_ratio'] > (1.8 * data['volatility_baseline']),
        (data['volatility_ratio'] >= (1.2 * data['volatility_baseline'])) & 
        (data['volatility_ratio'] <= (1.8 * data['volatility_baseline'])),
        data['volatility_ratio'] < (1.2 * data['volatility_baseline'])
    ]
    choices_vol = [2, 1, 0]  # 2=High, 1=Medium, 0=Low
    data['volatility_regime'] = np.select(conditions_vol, choices_vol, default=1)
    
    # Volume Momentum Dynamics
    # Volume acceleration signals
    data['volume_momentum_3d'] = (data['volume'] / data['volume'].shift(3)) - 1
    data['volume_momentum_5d'] = (data['volume'] / data['volume'].shift(5)) - 1
    data['combined_volume_acceleration'] = (data['volume_momentum_3d'] + data['volume_momentum_5d']) / 2
    
    # Volume regime classification
    conditions_vol_acc = [
        data['combined_volume_acceleration'] > 0.4,
        (data['combined_volume_acceleration'] >= 0.15) & (data['combined_volume_acceleration'] <= 0.4),
        data['combined_volume_acceleration'] < 0.15
    ]
    choices_vol_acc = [2, 1, 0]  # 2=High, 1=Medium, 0=Low
    data['volume_regime'] = np.select(conditions_vol_acc, choices_vol_acc, default=1)
    
    # Volume trend persistence
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_above_avg'] = (data['volume'] > data['volume_5d_avg']).astype(int)
    data['volume_persistence'] = data['volume_above_avg'].rolling(window=3, min_periods=2).sum()
    
    # Price Momentum Characteristics
    # Short-term price momentum
    data['price_momentum_2d'] = (data['close'] / data['close'].shift(2)) - 1
    data['price_momentum_3d'] = (data['close'] / data['close'].shift(3)) - 1
    data['combined_short_momentum'] = (data['price_momentum_2d'] + data['price_momentum_3d']) / 2
    
    # Medium-term price trend using linear regression
    def linear_slope(series, window):
        slopes = []
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(window)
                slope, _, _, _, _ = stats.linregress(x, y)
                slopes.append(slope)
        return pd.Series(slopes, index=series.index)
    
    data['price_slope_5d'] = linear_slope(data['close'], 5)
    data['price_slope_8d'] = linear_slope(data['close'], 8)
    
    # Momentum Divergence Detection
    # Price-volume divergence
    data['price_volume_divergence'] = np.abs(data['combined_short_momentum']) * np.abs(data['combined_volume_acceleration'])
    
    # Multi-timeframe consistency
    data['trend_consistency'] = ((data['price_slope_5d'] * data['price_slope_8d']) > 0).astype(int)
    
    # Volatility-adjusted momentum
    volatility_adjustments = [1.2, 1.0, 0.8]  # High, Medium, Low volatility
    data['volatility_adjustment'] = data['volatility_regime'].map(lambda x: volatility_adjustments[int(x)])
    data['volatility_adjusted_momentum'] = data['combined_short_momentum'] * data['volatility_adjustment']
    
    # Adaptive Alpha Generation
    # Base divergence signal
    data['base_divergence_signal'] = (data['price_volume_divergence'] * data['trend_consistency'] * 
                                     np.sign(data['combined_short_momentum']))
    
    # Regime-based multipliers
    regime_multipliers = {
        (2, 2): 2.5,  # High Vol + High Vol Acc
        (2, 1): 1.8,  # High Vol + Medium Vol Acc
        (2, 0): 1.2,  # High Vol + Low Vol Acc
        (1, 2): 1.5,  # Medium Vol + High Vol Acc
        (1, 1): 1.0,  # Medium Vol + Medium Vol Acc
        (1, 0): 0.6,  # Medium Vol + Low Vol Acc
        (0, 2): 0.8,  # Low Vol + High Vol Acc
        (0, 1): 0.4,  # Low Vol + Medium Vol Acc
        (0, 0): 0.2   # Low Vol + Low Vol Acc
    }
    
    data['regime_multiplier'] = data.apply(
        lambda row: regime_multipliers.get((row['volatility_regime'], row['volume_regime']), 1.0), 
        axis=1
    )
    
    # Final alpha factor with volume persistence as confidence filter
    data['alpha_factor'] = (data['base_divergence_signal'] * data['regime_multiplier'] * 
                           data['volatility_adjusted_momentum'] * 
                           (1 + 0.1 * data['volume_persistence']))
    
    return data['alpha_factor']
