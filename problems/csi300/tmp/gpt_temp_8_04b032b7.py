import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate market proxy (equal-weighted average of all stocks)
    # Note: In practice, this would need actual market data
    # Using average of all stocks as market proxy for demonstration
    market_close = data.groupby(level=0)['close'].mean()
    sector_close = market_close  # Using market as sector proxy for demonstration
    
    # Merge market and sector data
    data = data.join(market_close.rename('market_close'), how='left')
    data = data.join(sector_close.rename('sector_close'), how='left')
    
    # Calculate basic price changes
    data['close_ret_5'] = data['close'] / data.groupby(level=1)['close'].shift(5) - 1
    data['market_ret_5'] = data['market_close'] / data.groupby(level=0)['market_close'].shift(5) - 1
    data['sector_ret_5'] = data['sector_close'] / data.groupby(level=0)['sector_close'].shift(5) - 1
    
    # Stock vs. Market Gap
    data['stock_market_gap'] = (data['close_ret_5'] / (data['market_ret_5'] + 1e-6)) * \
                              (data['high'] - data['low']) / (abs(data['close'] - data.groupby(level=1)['close'].shift(2)) + 1e-6)
    
    # Gap Divergence
    data['gap_divergence'] = data['stock_market_gap'] - \
                            (data['close_ret_5'] / (data['sector_ret_5'] + 1e-6)) * \
                            (data['high'] - data['low']) / (abs(data['open'] - data.groupby(level=1)['close'].shift(2)) + 1e-6)
    
    # Volatility Adjustment
    data['volatility_adj'] = data['gap_divergence'] * \
                            (data['high'] - data['low']) / (abs(data['close'] - data['open']) + 1e-6)
    
    # Volume Scaling
    data['volume_scaling'] = data['volatility_adj'] * \
                            data['volume'] / (data.groupby(level=1)['volume'].shift(5) + 1e-6)
    
    # Gap Persistence
    gap_sign = np.sign(data['gap_divergence'])
    data['direction_persistence'] = 0
    for i in range(6):
        data['direction_persistence'] += (gap_sign == gap_sign.groupby(level=1).shift(i)).astype(float)
    data['direction_persistence'] = data['direction_persistence'] * \
                                   abs(data['gap_divergence']) / (data['high'] - data['low'] + 1e-6)
    
    # Asymmetry Shift
    data['asymmetry_shift'] = ((data['high'] - data['close']) - (data['close'] - data['low'])) / \
                             (data['high'] - data['low'] + 1e-6) * np.sign(data['gap_divergence'])
    
    # Volume Strength
    volume_ma = data.groupby(level=1)['volume'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    data['volume_strength'] = np.log(data['volume'] + 1e-6) / np.log(volume_ma + 1e-6) * \
                             abs((data['close'] - data.groupby(level=1)['close'].shift(1)) / \
                             (data.groupby(level=1)['close'].shift(1) + 1e-6)) * \
                             np.sign(data['close'] - data.groupby(level=1)['close'].shift(1))
    
    # Opening Efficiency
    data['opening_efficiency'] = ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-6)) * \
                                ((data['open'] - data.groupby(level=1)['close'].shift(1)) / \
                                (abs(data['open'] - data.groupby(level=1)['close'].shift(3)) + 1e-6))
    
    # Opening Momentum
    data['opening_momentum'] = (data['open'] - data.groupby(level=1)['close'].shift(1)) * \
                              (data['close'] - data['open']) * \
                              abs(data.groupby(level=1)['close'].shift(1) - data['open']) / \
                              (data['high'] - data['low'] + 1e-6)
    
    # Intraday Momentum
    mid_price = (data['high'] + data['low']) / 2
    data['intraday_momentum'] = np.log(data['high'] - data['low'] + 1e-6) / \
                               np.log(abs(data['close'] - mid_price) + 1e-6) * \
                               abs(data['close'] - mid_price) / (data['high'] - data['low'] + 1e-6)
    
    # Core Momentum
    data['core_momentum'] = data['opening_momentum'] * data['intraday_momentum'] * data['gap_divergence']
    
    # Regime Multipliers
    price_regime = np.where(data['asymmetry_shift'] > 0.3, 0.45, 
                           np.where(data['asymmetry_shift'] < -0.3, 0.18, 1.0))
    
    volume_regime = np.where(data['volume_strength'] > 1.4, 0.3,
                            np.where(data['volume_strength'] < 0.6, 0.1, 1.0))
    
    regime_multiplier = price_regime * volume_regime
    
    # Final Alpha
    data['alpha'] = data['core_momentum'] * \
                   (data['volume'] / (data.groupby(level=1)['volume'].shift(5) + 1e-6)) * \
                   data['direction_persistence'] * \
                   regime_multiplier
    
    # Return alpha series
    return data['alpha']
