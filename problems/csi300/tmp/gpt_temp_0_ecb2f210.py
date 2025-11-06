import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factors based on volatility-regime momentum, 
    liquidity-adjusted price inefficiency, gap mean reversion with volume,
    and intraday strength persistence.
    """
    # Make a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Calculate daily returns
    data['returns'] = data['close'].pct_change()
    
    # 1. Volatility-Regime Momentum
    # Calculate rolling volatility (5-day and 20-day)
    data['vol_5d'] = data['returns'].rolling(window=5).std()
    data['vol_20d'] = data['returns'].rolling(window=20).std()
    
    # Define volatility regimes
    vol_threshold = data['vol_20d'].rolling(window=60).median()
    high_vol_regime = data['vol_5d'] > vol_threshold
    
    # Volatility-regime momentum
    data['momentum_5d'] = data['close'].pct_change(5)
    data['momentum_20d'] = data['close'].pct_change(20)
    data['vol_regime_momentum'] = np.where(high_vol_regime, 
                                          data['momentum_5d'], 
                                          data['momentum_20d'])
    
    # 2. Liquidity-Adjusted Price Inefficiency
    # Price inefficiency: absolute return autocorrelation
    data['return_autocorr'] = data['returns'].rolling(window=20).apply(
        lambda x: x.autocorr(lag=1) if len(x) == 20 else np.nan, raw=False
    )
    data['price_inefficiency'] = np.abs(data['return_autocorr'])
    
    # Liquidity adjustment: volume concentration (Herfindahl index)
    data['volume_20d_avg'] = data['volume'].rolling(window=20).mean()
    data['volume_concentration'] = (data['volume'] / data['volume_20d_avg']) ** 2
    data['volume_concentration_20d'] = data['volume_concentration'].rolling(window=20).mean()
    
    # Liquidity-adjusted price inefficiency
    data['liquidity_adj_inefficiency'] = data['price_inefficiency'] * data['volume_concentration_20d']
    
    # 3. Gap Mean Reversion with Volume
    # Large opening gaps
    data['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Volume confirmation: current vs 20-day average
    data['volume_ratio'] = data['volume'] / data['volume_20d_avg']
    
    # Gap mean reversion factor (negative for mean reversion)
    gap_threshold = data['gap'].abs().rolling(window=60).quantile(0.7)
    large_gap = data['gap'].abs() > gap_threshold
    data['gap_mean_reversion'] = np.where(large_gap, -data['gap'] * data['volume_ratio'], 0)
    
    # 4. Intraday Strength Persistence
    # Intraday pattern: (Close - Open)/(High - Low)
    data['intraday_strength'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['intraday_strength'] = data['intraday_strength'].replace([np.inf, -np.inf], np.nan)
    
    # Volume persistence: volume autocorrelation lag 1
    data['volume_autocorr'] = data['volume'].rolling(window=20).apply(
        lambda x: x.autocorr(lag=1) if len(x) == 20 else np.nan, raw=False
    )
    
    # Intraday strength persistence
    data['intraday_persistence'] = data['intraday_strength'] * data['volume_autocorr']
    
    # Combine all factors (equal weighting for simplicity)
    factors = ['vol_regime_momentum', 'liquidity_adj_inefficiency', 
               'gap_mean_reversion', 'intraday_persistence']
    
    # Standardize each factor
    for factor in factors:
        data[f'{factor}_z'] = (data[factor] - data[factor].rolling(window=60).mean()) / data[factor].rolling(window=60).std()
    
    # Final combined factor (equal weighted z-scores)
    data['combined_factor'] = (data['vol_regime_momentum_z'] + 
                              data['liquidity_adj_inefficiency_z'] + 
                              data['gap_mean_reversion_z'] + 
                              data['intraday_persistence_z']) / 4
    
    return data['combined_factor']
