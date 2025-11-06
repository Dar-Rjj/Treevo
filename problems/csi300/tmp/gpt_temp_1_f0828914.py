import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Liquidity-Momentum Integration
    # Calculate daily effective spread
    data['mid_price'] = (data['high'] + data['low']) / 2
    data['effective_spread'] = 2 * np.abs(data['close'] - data['mid_price']) / data['mid_price']
    
    # Compute average trade size
    data['avg_trade_size'] = data['amount'] / data['volume']
    
    # Calculate returns
    data['ret_1d'] = data['close'].pct_change(1)
    data['ret_3d'] = data['close'].pct_change(3)
    data['ret_5d'] = data['close'].pct_change(5)
    
    # Combine momentum factors
    data['momentum_combined'] = (data['ret_1d'] + data['ret_3d'] + data['ret_5d']) / 3
    
    # Trade size concentration (z-score normalized)
    data['trade_size_z'] = (data['avg_trade_size'] - data['avg_trade_size'].rolling(window=20, min_periods=10).mean()) / data['avg_trade_size'].rolling(window=20, min_periods=10).std()
    
    # Liquidity-momentum factor
    data['liquidity_momentum'] = data['momentum_combined'] * (1 - data['effective_spread']) * data['trade_size_z']
    
    # Price-Volume Microstructure Divergence
    # Price change vs volume change correlation (5-day rolling)
    data['price_change'] = data['close'].pct_change(1)
    data['volume_change'] = data['volume'].pct_change(1)
    data['price_volume_corr'] = data['price_change'].rolling(window=5, min_periods=3).corr(data['volume_change'])
    
    # Intraday price efficiency
    data['price_efficiency'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volume persistence (5-day EMA of volume)
    data['volume_persistence'] = data['volume'].ewm(span=5, adjust=False).mean()
    
    # Microstructure divergence factor
    data['microstructure_div'] = data['price_volume_corr'] * data['price_efficiency'] * data['volume_persistence']
    
    # Volatility-Regime Momentum
    # 5-day realized volatility (Parkinson estimator)
    data['log_hl'] = np.log(data['high'] / data['low'])
    data['realized_vol'] = np.sqrt((1 / (4 * np.log(2))) * (data['log_hl'] ** 2).rolling(window=5, min_periods=3).mean())
    
    # Volatility regime classification
    data['vol_regime'] = pd.cut(data['realized_vol'], 
                               bins=[0, data['realized_vol'].quantile(0.33), data['realized_vol'].quantile(0.66), np.inf],
                               labels=[0, 1, 2])
    
    # Momentum across regimes (weighted by volatility persistence)
    vol_persistence = data['realized_vol'].ewm(span=5, adjust=False).mean()
    data['regime_momentum'] = data['momentum_combined'] * (data['vol_regime'].astype(float) + 1) * vol_persistence
    
    # Order Flow Imbalance
    # Trade size skewness (3-day rolling)
    data['trade_size_skew'] = data['avg_trade_size'].rolling(window=3, min_periods=2).apply(
        lambda x: x.skew() if len(x) > 1 else np.nan
    )
    
    # Price discovery speed
    data['discovery_speed'] = np.abs(data['close'] - data['mid_price'])
    
    # Flow persistence (EMA of amount)
    data['flow_persistence'] = data['amount'].ewm(span=3, adjust=False).mean()
    
    # Order flow imbalance factor
    data['order_flow_imbalance'] = data['trade_size_skew'] * data['discovery_speed'] * data['flow_persistence']
    
    # Fractal Efficiency with Liquidity
    # Local fractal dimension (5-day Hurst-like measure)
    def hurst_like(series):
        if len(series) < 5:
            return np.nan
        lags = range(2, 5)
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    data['fractal_dim'] = data['close'].rolling(window=5, min_periods=3).apply(hurst_like, raw=False)
    
    # Liquidity-adjusted efficiency
    data['liquidity_adj_efficiency'] = data['price_efficiency'] / (data['effective_spread'] + 1e-6)
    
    # Volatility adjustment
    vol_adj = 1 / (data['realized_vol'] + 1e-6)
    
    # Fractal efficiency factor
    data['fractal_efficiency'] = data['fractal_dim'] * data['liquidity_adj_efficiency'] * vol_adj
    
    # Combine all factors with equal weighting
    factors = ['liquidity_momentum', 'microstructure_div', 'regime_momentum', 
               'order_flow_imbalance', 'fractal_efficiency']
    
    # Normalize each factor by its rolling z-score
    combined_factor = pd.Series(0, index=data.index)
    for factor in factors:
        factor_series = data[factor]
        rolling_mean = factor_series.rolling(window=20, min_periods=10).mean()
        rolling_std = factor_series.rolling(window=20, min_periods=10).std()
        normalized_factor = (factor_series - rolling_mean) / rolling_std
        combined_factor += normalized_factor
    
    # Final factor (equally weighted combination)
    final_factor = combined_factor / len(factors)
    
    return final_factor
