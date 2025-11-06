import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Intraday Momentum Divergence
    intraday_range = data['high'] - data['low']
    overnight_gap = data['close'] - data['open']
    
    range_sign = np.sign(intraday_range.diff())
    gap_sign = np.sign(overnight_gap.diff())
    divergence = (range_sign != gap_sign).astype(int)
    
    volume_strength = data['volume'] / data['volume'].rolling(window=20, min_periods=1).mean()
    alpha1 = divergence * volume_strength
    
    # High-Low Range Compression Breakout
    prev_close = data['close'].shift(1)
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - prev_close)
    tr3 = abs(data['low'] - prev_close)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    tr_ratio = true_range / true_range.rolling(window=20, min_periods=1).mean()
    vol_breakout = data['volume'] / data['volume'].rolling(window=20, min_periods=1).median()
    alpha2 = tr_ratio * vol_breakout
    
    # Volatility-Regressed Return Momentum
    returns_5d = data['close'].pct_change(periods=5)
    volatility = ((data['high'] - data['low']) / data['close']).rolling(window=20, min_periods=1).mean()
    
    def rolling_beta(x, y, window):
        betas = []
        for i in range(len(x)):
            if i < window - 1:
                betas.append(np.nan)
            else:
                x_window = x.iloc[i-window+1:i+1]
                y_window = y.iloc[i-window+1:i+1]
                if len(x_window.dropna()) >= 5 and len(y_window.dropna()) >= 5:
                    beta = np.cov(x_window, y_window)[0,1] / np.var(x_window)
                    betas.append(beta)
                else:
                    betas.append(np.nan)
        return pd.Series(betas, index=x.index)
    
    beta_returns_vol = rolling_beta(returns_5d, volatility, 20)
    inverse_beta = 1 / (abs(beta_returns_vol) + 1e-6)
    alpha3 = returns_5d * inverse_beta
    
    # Price-Volume Trend Convergence
    price_ema_5 = data['close'].ewm(span=5, adjust=False).mean()
    price_ema_10 = data['close'].ewm(span=10, adjust=False).mean()
    price_ema_20 = data['close'].ewm(span=20, adjust=False).mean()
    
    volume_ema_5 = data['volume'].ewm(span=5, adjust=False).mean()
    volume_ema_10 = data['volume'].ewm(span=10, adjust=False).mean()
    volume_ema_20 = data['volume'].ewm(span=20, adjust=False).mean()
    
    price_convergence = ((price_ema_5 > price_ema_10) & (price_ema_10 > price_ema_20)).astype(int) - \
                       ((price_ema_5 < price_ema_10) & (price_ema_10 < price_ema_20)).astype(int)
    
    volume_convergence = ((volume_ema_5 > volume_ema_10) & (volume_ema_10 > volume_ema_20)).astype(int) - \
                        ((volume_ema_5 < volume_ema_10) & (volume_ema_10 < volume_ema_20)).astype(int)
    
    alpha4 = price_convergence * volume_convergence
    
    # Overnight Gap Mean Reversion Strength
    overnight_gaps = (data['open'] - prev_close) / prev_close
    
    gap_direction = np.sign(overnight_gaps)
    consecutive_gaps = gap_direction.groupby((gap_direction != gap_direction.shift(1)).cumsum()).cumcount() + 1
    
    vol_participation = data['volume'] / data['volume'].rolling(window=10, min_periods=1).mean()
    
    alpha5 = overnight_gaps * consecutive_gaps * vol_participation
    
    # Combine all alphas with equal weights
    alpha_combined = (alpha1.fillna(0) + alpha2.fillna(0) + alpha3.fillna(0) + 
                     alpha4.fillna(0) + alpha5.fillna(0)) / 5
    
    return alpha_combined
