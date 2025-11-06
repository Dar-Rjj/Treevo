import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    import pandas as pd
    import numpy as np
    
    # 1. Multi-timeframe momentum confirmation
    mom_1d = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    mom_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    
    # Momentum consistency: correlation between short and medium term
    mom_correlation = mom_1d.rolling(window=5).corr(mom_3d)
    momentum_confirmed = (mom_1d + mom_3d) * (1 + mom_correlation)
    
    # 2. Dynamic volatility adjustment
    true_range = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': abs(df['high'] - df['close'].shift(1)),
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    volatility_5d = true_range.rolling(window=5).mean() / df['close']
    volatility_regime = volatility_5d / volatility_5d.rolling(window=20).mean()
    
    # 3. Volume-price divergence detection
    volume_trend = df['volume'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] / np.mean(x)
    )
    price_trend = df['close'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] / np.mean(x)
    )
    volume_price_divergence = volume_trend * price_trend * np.sign(volume_trend - price_trend)
    
    # 4. Range efficiency with momentum
    daily_range = df['high'] - df['low']
    range_efficiency = (df['close'] - df['open']) / (daily_range + 1e-7)
    range_momentum = range_efficiency * mom_1d
    
    # 5. Smart money flow detection
    avg_trade_size = df['amount'] / (df['volume'] + 1e-7)
    trade_size_momentum = (avg_trade_size - avg_trade_size.shift(3)) / (avg_trade_size.shift(3) + 1e-7)
    price_vs_trade_size = mom_1d * trade_size_momentum
    
    # 6. Adaptive weighting based on volatility regime
    high_vol_weight = np.where(volatility_regime > 1.2, 1.0, volatility_regime / 1.2)
    low_vol_weight = 1 - high_vol_weight
    
    # High volatility: emphasize range and volume factors
    high_vol_component = (0.4 * range_momentum + 
                         0.35 * volume_price_divergence + 
                         0.25 * price_vs_trade_size)
    
    # Low volatility: emphasize momentum factors  
    low_vol_component = (0.5 * momentum_confirmed + 
                        0.3 * range_momentum + 
                        0.2 * volume_price_divergence)
    
    # Final adaptive combination
    factor = (high_vol_weight * high_vol_component + 
             low_vol_weight * low_vol_component)
    
    return factor
