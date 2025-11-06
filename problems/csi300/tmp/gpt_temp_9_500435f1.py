import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['TrueRange'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Volatility Asymmetry Microstructure
    # Gap-Intraday Alignment
    data['gap_intraday'] = np.sign((data['open'] - data['prev_close']) * 
                                  (data['close'] - data['open']) * 
                                  (data['TrueRange'] ** 2))
    
    # Volume-Volatility Asymmetry
    epsilon = 1e-8
    data['volume_vol_ratio'] = data['volume'] / (data['amount'] + epsilon)
    data['vol_vol_asy_up'] = data['volume_vol_ratio'] * data['TrueRange']
    data['vol_vol_asy_down'] = data['volume_vol_ratio'] * data['TrueRange']
    
    # Apply conditions for up/down days
    up_condition = data['close'] > data['prev_close']
    down_condition = data['close'] < data['prev_close']
    
    data['VolumeVolatilityAsymmetry'] = np.where(
        down_condition, 
        data['vol_vol_asy_up'].shift(1) / (data['vol_vol_asy_down'] + epsilon),
        data['vol_vol_asy_up'] / (data['vol_vol_asy_down'].shift(1) + epsilon)
    )
    
    # Range Bias
    data['RangeBias'] = ((data['high'] - data['close']) - 
                        (data['close'] - data['low'])) * \
                       data['TrueRange'] / (data['high'] - data['low'] + epsilon)
    
    # Multi-Timeframe Memory
    # Volatility Bias Momentum
    data['VolBiasMomentum'] = data['RangeBias'] - data['RangeBias'].shift(5)
    
    # Volume-Volatility Trend
    data['VolVolTrend'] = data['VolumeVolatilityAsymmetry'] / \
                         (data['VolumeVolatilityAsymmetry'].shift(20) + epsilon) - 1
    
    # Liquidity-Volatility Regime
    # Volatility Concentration
    vol_concentration = []
    vol_metric = data['volume_vol_ratio'] * data['TrueRange']
    threshold = vol_metric.rolling(window=10, min_periods=1).mean()
    
    for i in range(len(data)):
        if i < 9:
            vol_concentration.append(0)
        else:
            window_data = vol_metric.iloc[i-9:i+1]
            window_threshold = threshold.iloc[i]
            above_count = (window_data > window_threshold).sum()
            below_count = (window_data < window_threshold).sum()
            vol_concentration.append(above_count - below_count)
    
    data['VolatilityConcentration'] = vol_concentration
    
    # Trade Size Momentum
    data['trade_size_vol'] = (data['amount'] / (data['volume'] + epsilon)) * data['TrueRange']
    data['TradeSizeMomentum'] = data['trade_size_vol'] / \
                               (data['trade_size_vol'].shift(5) + epsilon) - 1
    
    # Adaptive Synthesis
    # High Regime
    data['HighRegime'] = data['RangeBias'] * data['trade_size_vol']
    
    # Low Regime
    data['LowRegime'] = data['RangeBias'] / (data['trade_size_vol'] + epsilon)
    
    # Final Factor - Weighted combination using Trade Size Momentum
    # Use absolute value of TradeSizeMomentum as weight, normalized
    weight = abs(data['TradeSizeMomentum'])
    weight_normalized = weight / (weight.rolling(window=20, min_periods=1).mean() + epsilon)
    
    # Combine regimes based on weight
    data['FinalFactor'] = weight_normalized * data['HighRegime'] + \
                         (1 - weight_normalized) * data['LowRegime']
    
    # Clean up intermediate columns
    result = data['FinalFactor'].copy()
    
    return result
