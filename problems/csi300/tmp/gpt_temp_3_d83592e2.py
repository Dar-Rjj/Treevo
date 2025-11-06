import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily volatility for stock
    stock_vol = (df['high'] - df['low']) / df['close'].shift(1)
    
    # Calculate daily volatility for index (using market average as proxy)
    index_vol = (df['high'].rolling(window=5).mean() - df['low'].rolling(window=5).mean()) / df['close'].shift(1).rolling(window=5).mean()
    
    # Calculate sector volatility (using rolling window as sector proxy)
    sector_vol = stock_vol.rolling(window=5).mean()
    
    # Calculate volatility correlations
    vol_corr_sector = stock_vol.rolling(window=5).corr(sector_vol)
    
    # Calculate transmission lag and strength
    transmission_lag = pd.Series(index=df.index, dtype=float)
    transmission_strength = pd.Series(index=df.index, dtype=float)
    
    for i in range(2, len(df)):
        current_data = []
        for k in [0, 2]:
            if i - k - 1 >= 0:
                corr_window = stock_vol.iloc[max(0, i-4):i+1]
                index_vol_lagged = index_vol.shift(k).iloc[max(0, i-4):i+1]
                if len(corr_window) >= 3 and len(index_vol_lagged) >= 3:
                    corr_val = corr_window.corr(index_vol_lagged)
                    current_data.append((k, corr_val if not pd.isna(corr_val) else 0))
        
        if current_data:
            best_k, best_corr = max(current_data, key=lambda x: abs(x[1]))
            transmission_lag.iloc[i] = best_k
            transmission_strength.iloc[i] = best_corr
    
    # Calculate flow components
    price_range = df['high'] - df['low']
    price_range = price_range.replace(0, np.nan)
    
    buy_pressure = ((df['close'] - df['low']) / price_range) * df['volume']
    sell_pressure = ((df['high'] - df['close']) / price_range) * df['volume']
    
    # Calculate flow momentum
    inflow_momentum = (buy_pressure - buy_pressure.shift(1)) / buy_pressure.shift(1).replace(0, np.nan)
    outflow_momentum = (sell_pressure - sell_pressure.shift(1)) / sell_pressure.shift(1).replace(0, np.nan)
    
    # Calculate volatility-flow coupling
    inflow_vol_sensitivity = inflow_momentum / index_vol.replace(0, np.nan)
    outflow_vol_sensitivity = outflow_momentum / index_vol.replace(0, np.nan)
    
    # Calculate transmission efficiency adjustments
    lag_adjusted_inflow = inflow_momentum * transmission_lag
    strength_adjusted_outflow = outflow_momentum * transmission_strength
    
    # Calculate sector flow divergence (using rolling mean as sector proxy)
    sector_inflow_avg = inflow_momentum.rolling(window=5).mean()
    sector_outflow_avg = outflow_momentum.rolling(window=5).mean()
    
    relative_inflow = inflow_momentum - sector_inflow_avg
    relative_outflow = outflow_momentum - sector_outflow_avg
    
    # Calculate market flow divergence (using index volatility as market proxy)
    market_inflow_divergence = inflow_momentum - inflow_momentum.rolling(window=5).mean()
    market_outflow_divergence = outflow_momentum - outflow_momentum.rolling(window=5).mean()
    
    # Calculate composite factors
    core_transmission_factor = (relative_inflow * inflow_vol_sensitivity) - (relative_outflow * strength_adjusted_outflow)
    
    flow_divergence_signal = np.sign(market_inflow_divergence + market_outflow_divergence + relative_inflow + relative_outflow)
    
    # Final composite factor
    composite_factor = core_transmission_factor * flow_divergence_signal
    
    return composite_factor.fillna(0)
