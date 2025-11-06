import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Market Microstructure Regime Detection factor
    Detects regime shifts in market microstructure using price, volume, and amount data
    """
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Price Discovery Efficiency Regimes
    # Intraday price variance to overnight gap ratio
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_range'] = (data['high'] - data['low']) / data['open']
    data['price_discovery_ratio'] = data['intraday_range'] / (abs(data['overnight_gap']) + 1e-8)
    
    # Price impact asymmetry across volume regimes
    data['price_change'] = (data['close'] - data['open']) / data['open']
    data['volume_zscore'] = (data['volume'] - data['volume'].rolling(window=20).mean()) / data['volume'].rolling(window=20).std()
    data['high_volume_regime'] = (data['volume_zscore'] > 1).astype(int)
    data['low_volume_regime'] = (data['volume_zscore'] < -1).astype(int)
    
    # Price impact in different volume regimes
    data['price_impact_high_vol'] = data['price_change'] * data['high_volume_regime']
    data['price_impact_low_vol'] = data['price_change'] * data['low_volume_regime']
    
    # 2. Order Flow Imbalance Dynamics
    # Micro-scale order flow clustering using amount/volume ratio
    data['avg_trade_size'] = data['amount'] / (data['volume'] + 1e-8)
    data['trade_size_volatility'] = data['avg_trade_size'].rolling(window=10).std()
    data['order_clustering'] = data['avg_trade_size'] / (data['trade_size_volatility'] + 1e-8)
    
    # Large trade absorption capacity
    data['large_trade_threshold'] = data['avg_trade_size'].rolling(window=20).quantile(0.8)
    data['large_trade_days'] = (data['avg_trade_size'] > data['large_trade_threshold']).astype(int)
    data['large_trade_absorption'] = data['large_trade_days'] * (1 - abs(data['price_change']))
    
    # 3. Liquidity Provision Regime Shifts
    # Bid-ask spread proxy using high-low range relative to price
    data['spread_proxy'] = (data['high'] - data['low']) / data['close']
    data['spread_volatility'] = data['spread_proxy'].rolling(window=10).std()
    
    # Depth resilience during high volatility
    data['volatility_regime'] = (data['spread_proxy'] > data['spread_proxy'].rolling(window=20).quantile(0.7)).astype(int)
    data['volume_resilience'] = data['volume'] * (1 - data['volatility_regime'])
    
    # 4. Market Maker Behavior Regimes
    # Inventory management patterns using price reversal signals
    data['overnight_return'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    data['reversal_signal'] = np.sign(data['overnight_return']) * np.sign(data['intraday_return'])
    
    # Quote revision frequency proxy using intraday price oscillations
    data['price_oscillations'] = ((data['high'] - data['open']) + (data['open'] - data['low'])) / data['open']
    
    # 5. Information Asymmetry Regime Detection
    # Adverse selection cost patterns
    data['adverse_selection'] = abs(data['price_change']) / (data['volume'] + 1e-8)
    
    # Informed trading concentration in volume buckets
    data['volume_quantile'] = data['volume'].rolling(window=20).apply(
        lambda x: pd.qcut(x, 4, labels=False, duplicates='drop').iloc[-1] if len(x) == 20 else np.nan
    )
    data['informed_trading_concentration'] = data['adverse_selection'] * (data['volume_quantile'] == 3).astype(float)
    
    # Combine all regime signals into final factor
    factor = (
        data['price_discovery_ratio'].rolling(window=5).mean() * 0.15 +
        (data['price_impact_high_vol'] - data['price_impact_low_vol']).rolling(window=5).mean() * 0.15 +
        data['order_clustering'].rolling(window=5).mean() * 0.15 +
        data['large_trade_absorption'].rolling(window=5).mean() * 0.15 +
        (1 / (data['spread_volatility'] + 1e-8)).rolling(window=5).mean() * 0.15 +
        data['volume_resilience'].rolling(window=5).mean() * 0.10 +
        data['reversal_signal'].rolling(window=5).mean() * 0.075 +
        data['price_oscillations'].rolling(window=5).mean() * 0.05 +
        data['informed_trading_concentration'].rolling(window=5).mean() * 0.05
    )
    
    return factor
