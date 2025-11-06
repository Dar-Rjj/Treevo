import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum with Liquidity Scaling factor
    Combines momentum and mean reversion signals with volatility regime detection
    and liquidity-based scaling for improved predictive power.
    """
    data = df.copy()
    
    # Momentum Component
    # Short-term Momentum
    data['ret_5d'] = data['close'] / data['close'].shift(5) - 1
    data['ret_3d'] = data['close'] / data['close'].shift(3) - 1
    
    # Medium-term Momentum
    data['ret_20d'] = data['close'] / data['close'].shift(20) - 1
    data['ret_10d'] = data['close'] / data['close'].shift(10) - 1
    
    # Momentum Acceleration
    data['mom_accel_1'] = data['ret_5d'] - data['ret_10d']
    data['mom_accel_2'] = data['ret_3d'] - data['ret_20d']
    
    # Mean Reversion Component
    # Price Deviation from Recent Range
    data['high_5d'] = data['high'].rolling(window=5).max()
    data['low_5d'] = data['low'].rolling(window=5).min()
    data['price_dev_5d'] = (data['close'] - data['low_5d']) / (data['high_5d'] - data['low_5d'])
    
    data['high_10d'] = data['high'].rolling(window=10).max()
    data['low_10d'] = data['low'].rolling(window=10).min()
    data['price_dev_10d'] = (data['close'] - data['low_10d']) / (data['high_10d'] - data['low_10d'])
    
    # Intraday Reversal Signal
    data['intraday_signal'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['overnight_gap'] = data['open'] / data['close'].shift(1) - 1
    
    # Volume-Weighted Mean Reversion
    data['volume_pct_5d'] = data['volume'].rolling(window=5).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
    )
    data['vw_price_dev'] = data['price_dev_5d'] * data['volume_pct_5d']
    
    data['volume_trend'] = data['volume'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0
    )
    data['scaled_intraday'] = data['intraday_signal'] * data['volume_trend']
    
    # Volatility Regime Detection
    # ATR Calculation
    def calculate_atr(high, low, close, window):
        tr = np.maximum(high - low, 
                       np.maximum(abs(high - close.shift(1)), 
                                 abs(low - close.shift(1))))
        return tr.rolling(window=window).mean()
    
    data['atr_20d'] = calculate_atr(data['high'], data['low'], data['close'], 20)
    data['atr_60d'] = calculate_atr(data['high'], data['low'], data['close'], 60)
    
    # Regime Classification
    data['high_vol_regime'] = data['atr_20d'] > data['atr_60d']
    
    # Liquidity Adjustment
    # Dollar Volume Calculation
    data['dollar_volume'] = data['close'] * data['volume']
    data['avg_dollar_volume_20d'] = data['dollar_volume'].rolling(window=20).mean()
    
    # Final Factor Assembly
    # Combine momentum signals (equal weighting)
    momentum_signal = (data['ret_5d'] + data['ret_3d'] + data['ret_20d'] + 
                      data['ret_10d'] + data['mom_accel_1'] + data['mom_accel_2']) / 6
    
    # Combine mean reversion signals (equal weighting)
    mean_reversion_signal = (data['price_dev_5d'] + data['price_dev_10d'] + 
                            data['intraday_signal'] + data['overnight_gap'] + 
                            data['vw_price_dev'] + data['scaled_intraday']) / 6
    
    # Apply volatility regime weighting
    combined_signal = np.where(
        data['high_vol_regime'],
        # High volatility: 70% mean reversion, 30% momentum
        0.7 * mean_reversion_signal + 0.3 * momentum_signal,
        # Low volatility: 30% mean reversion, 70% momentum
        0.3 * mean_reversion_signal + 0.7 * momentum_signal
    )
    
    # Apply liquidity scaling with square root transformation
    liquidity_scaling = np.sqrt(data['avg_dollar_volume_20d'])
    final_factor = combined_signal / liquidity_scaling
    
    # Return factor series
    return pd.Series(final_factor, index=data.index, name='regime_adaptive_momentum_liquidity')
