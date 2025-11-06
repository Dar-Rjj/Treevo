import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Path Efficiency Components
    # Intraday Efficiency
    intraday_eff = (data['close'] - data['open']) / (data['high'] - data['low'])
    intraday_eff = intraday_eff.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Multi-day Efficiency (3-day and 5-day)
    def multi_day_efficiency(window):
        if window < 2:
            return pd.Series(index=data.index, data=0)
        
        price_change = data['close'] - data['close'].shift(window-1)
        total_volatility = 0
        for i in range(window):
            total_volatility += abs(data['close'].shift(i) - data['close'].shift(i+1))
        
        efficiency = price_change / total_volatility
        return efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    eff_3day = multi_day_efficiency(3)
    eff_5day = multi_day_efficiency(5)
    
    # Volatility-Adjusted Efficiency
    volatility = data['high'] - data['low']
    vol_adj_intraday_eff = intraday_eff / (volatility + 1e-8)
    vol_adj_intraday_eff = vol_adj_intraday_eff.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Volume Confirmation Signals
    # Normalize volume using rolling z-score (20-day window)
    volume_zscore = (data['volume'] - data['volume'].rolling(window=20).mean()) / (data['volume'].rolling(window=20).std() + 1e-8)
    
    # High Efficiency + Low Volume (exhaustion signal)
    high_eff_low_vol = intraday_eff * (1 - volume_zscore.rank(pct=True))
    
    # Low Efficiency + High Volume (accumulation signal)
    low_eff_high_vol = (1 - abs(intraday_eff).rank(pct=True)) * volume_zscore.rank(pct=True)
    
    # Multi-Timeframe Integration
    # 1-day: Immediate price efficiency
    day1_factor = intraday_eff
    
    # 3-day: Short-term efficiency trend
    eff_3day_ma = eff_3day.rolling(window=3).mean()
    day3_factor = eff_3day_ma * np.sign(eff_3day)
    
    # 5-day: Medium-term efficiency persistence
    eff_5day_trend = eff_5day.rolling(window=5).apply(lambda x: 1 if (x > 0).sum() > (x < 0).sum() else -1)
    day5_factor = eff_5day * eff_5day_trend
    
    # Combine all components with weights
    volume_confirmation = high_eff_low_vol - low_eff_high_vol
    
    # Final factor calculation
    factor = (
        0.4 * day1_factor.rank(pct=True) +
        0.3 * day3_factor.rank(pct=True) + 
        0.2 * day5_factor.rank(pct=True) +
        0.1 * volume_confirmation.rank(pct=True)
    )
    
    # Normalize the final factor
    factor = (factor - factor.rolling(window=20).mean()) / (factor.rolling(window=20).std() + 1e-8)
    
    return factor
