import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Asset Microstructure Momentum Synthesis factor that combines:
    - Volatility transmission signals
    - Spread-regime adjusted momentum
    - Cross-timeframe momentum divergence
    - Volume-price efficiency momentum
    - Order flow imbalance momentum
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Volatility Transmission Component
    # Calculate rolling volatility with 20-day window
    vol_20d = data['close'].pct_change().rolling(window=20).std()
    vol_5d = data['close'].pct_change().rolling(window=5).std()
    
    # Volatility transmission intensity (lead-lag relationship)
    vol_transmission = (vol_5d / vol_20d.rolling(window=5).mean()) - 1
    
    # 2. Microstructure Liquidity Momentum
    # Estimate bid-ask spread using high-low range as proxy
    daily_range = (data['high'] - data['low']) / data['close']
    spread_regime = daily_range.rolling(window=20).apply(
        lambda x: pd.qcut(x, 3, labels=False, duplicates='drop').iloc[-1] 
        if len(x.dropna()) >= 15 else np.nan, raw=False
    )
    
    # Price momentum adjusted for spread regime
    momentum_5d = data['close'] / data['close'].shift(5) - 1
    momentum_10d = data['close'] / data['close'].shift(10) - 1
    
    # Spread-regime adjusted momentum
    spread_adjusted_momentum = momentum_5d * (1 - spread_regime / 3)
    
    # 3. Cross-Timeframe Momentum Divergence
    momentum_divergence = (momentum_5d - momentum_10d) / (np.abs(momentum_5d) + np.abs(momentum_10d) + 1e-8)
    
    # 4. Volume-Price Efficiency Momentum
    # Calculate volume-price efficiency (price change per unit volume)
    price_change = data['close'].pct_change()
    volume_efficiency = price_change / (data['volume'].rolling(window=5).mean() + 1e-8)
    
    # Efficiency regime classification
    efficiency_zscore = volume_efficiency.rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8) if len(x.dropna()) >= 15 else np.nan, 
        raw=False
    )
    
    efficiency_regime = np.where(efficiency_zscore > 1, 2, 
                                np.where(efficiency_zscore < -1, 0, 1))
    
    # Efficiency-weighted momentum
    efficiency_weighted_momentum = momentum_5d * (efficiency_regime / 2)
    
    # 5. Order Flow Imbalance Momentum
    # Estimate order flow using amount/volume ratio and price movement
    price_trend = data['close'].rolling(window=5).mean() / data['close'].rolling(window=10).mean() - 1
    volume_trend = data['volume'].rolling(window=5).mean() / data['volume'].rolling(window=10).mean() - 1
    
    # Order flow imbalance proxy
    flow_imbalance = price_trend * volume_trend
    
    # Flow-enhanced momentum
    flow_enhanced_momentum = momentum_5d * (1 + flow_imbalance)
    
    # Combine all components with equal weighting
    volatility_component = vol_transmission.rolling(window=5).mean()
    liquidity_component = spread_adjusted_momentum
    timeframe_component = momentum_divergence
    efficiency_component = efficiency_weighted_momentum
    flow_component = flow_enhanced_momentum
    
    # Normalize each component by its rolling standard deviation
    def zscore_normalize(series, window=20):
        return (series - series.rolling(window=window).mean()) / (series.rolling(window=window).std() + 1e-8)
    
    vol_norm = zscore_normalize(volatility_component)
    liq_norm = zscore_normalize(liquidity_component)
    time_norm = zscore_normalize(timeframe_component)
    eff_norm = zscore_normalize(efficiency_component)
    flow_norm = zscore_normalize(flow_component)
    
    # Final factor combination
    final_factor = (
        0.2 * vol_norm + 
        0.2 * liq_norm + 
        0.2 * time_norm + 
        0.2 * eff_norm + 
        0.2 * flow_norm
    )
    
    return final_factor
