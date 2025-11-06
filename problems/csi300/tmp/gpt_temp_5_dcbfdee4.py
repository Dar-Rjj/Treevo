import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Asymmetric Volatility Momentum Factor
    # Calculate directional volatility components
    df['upside_move'] = np.maximum(0, df['high'] - df['close'])
    df['downside_move'] = np.maximum(0, df['close'] - df['low'])
    
    df['upside_vol'] = df['upside_move'].rolling(window=5).std()
    df['downside_vol'] = df['downside_move'].rolling(window=5).std()
    
    # Compute volatility asymmetry ratio
    df['vol_asymmetry'] = df['upside_vol'] / df['downside_vol']
    df['vol_asymmetry_smooth'] = df['vol_asymmetry'].rolling(window=3).mean()
    
    # Combine with price momentum
    df['price_return_5d'] = df['close'].pct_change(5)
    df['raw_vol'] = df['close'].pct_change().rolling(window=5).std()
    asym_momentum = df['price_return_5d'] * df['vol_asymmetry_smooth'] / df['raw_vol']
    
    # Multi-Timeframe Volume-Price Alignment Factor
    # Short-term alignment
    df['volume_change'] = df['volume'].pct_change()
    df['daily_return'] = df['close'].pct_change()
    
    # 3-day price-volume correlation
    def rolling_corr(x, y, window):
        return x.rolling(window).corr(y)
    
    df['pv_corr_3d'] = rolling_corr(df['daily_return'], df['volume_change'], 3)
    
    # Direction consistency and strength
    df['direction_consistency'] = np.sign(df['daily_return']) * np.sign(df['volume_change'])
    df['strength'] = abs(df['daily_return']) * abs(df['volume_change'])
    df['short_term_align'] = df['pv_corr_3d'] * df['direction_consistency'] * df['strength']
    
    # Medium-term alignment
    df['price_trend_10d'] = df['close'].pct_change(10)
    
    # Volume trend (10-day slope)
    def volume_slope(series, window):
        x = np.arange(window)
        def calc_slope(y):
            if len(y) == window and not y.isna().any():
                return np.polyfit(x, y, 1)[0]
            return np.nan
        return series.rolling(window).apply(calc_slope, raw=False)
    
    df['volume_trend_10d'] = volume_slope(df['volume'], 10)
    df['medium_term_align'] = np.sign(df['price_trend_10d']) * np.sign(df['volume_trend_10d'])
    
    # Cross-timeframe integration
    df['alignment_diff'] = df['short_term_align'] - df['medium_term_align']
    df['alignment_combined'] = df['short_term_align'] * df['medium_term_align']
    
    # Liquidity-Adjusted Extreme Reversal Factor
    df['daily_vol'] = df['close'].pct_change().rolling(window=20).std()
    df['extreme_move'] = abs(df['daily_return']) > (2 * df['daily_vol'])
    
    # Consecutive same-direction moves
    df['return_sign'] = np.sign(df['daily_return'])
    df['consecutive_days'] = df['return_sign'].groupby((df['return_sign'] != df['return_sign'].shift()).cumsum()).cumcount() + 1
    df['consecutive_extreme'] = (df['consecutive_days'] >= 3) & df['extreme_move']
    
    # Gap events
    df['gap_size'] = abs(df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['gap_event'] = df['gap_size'] > df['daily_vol']
    
    # Liquidity conditions
    df['volume_ma_5d'] = df['volume'].rolling(window=5).mean()
    df['volume_relative'] = df['volume'] / df['volume_ma_5d']
    df['amount_per_trade'] = df['amount'] / df['volume']
    
    # Liquidity score
    df['liquidity_score'] = df['volume_relative'] * (1 / (1 + abs(df['amount_per_trade'].pct_change())))
    
    # Reversal probability
    extreme_conditions = df['extreme_move'] | df['consecutive_extreme'] | df['gap_event']
    df['reversal_prob'] = np.where(extreme_conditions, 
                                  df['liquidity_score'] * df['daily_vol'], 
                                  0)
    
    # Volume-Weighted Price Efficiency Factor
    # Intraday efficiency
    df['price_range'] = df['high'] - df['low']
    df['range_utilization'] = (df['close'] - df['low']) / df['price_range'].replace(0, np.nan)
    
    # Volume concentration (simplified)
    df['volume_efficiency'] = 1 - abs(df['range_utilization'] - 0.5) * 2
    df['intraday_efficiency'] = df['range_utilization'] * df['volume_efficiency']
    
    # Interday efficiency
    df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['intraday_move'] = (df['close'] - df['open']) / df['open']
    df['gap_fill_ratio'] = -df['overnight_gap'] / df['intraday_move'].replace(0, np.nan)
    df['gap_efficiency'] = 1 - abs(df['gap_fill_ratio'])
    
    # Multi-day efficiency consistency
    df['efficiency_consistency'] = df['intraday_efficiency'].rolling(window=5).std()
    
    # Volume-weighted combination
    df['volume_weight'] = df['volume'] / df['volume'].rolling(window=10).mean()
    composite_efficiency = (df['intraday_efficiency'] * 0.4 + 
                          df['gap_efficiency'] * 0.3 + 
                          (1 - df['efficiency_consistency']) * 0.3) * df['volume_weight']
    
    # Final factor combination with weights
    factor = (asym_momentum * 0.25 + 
             df['alignment_combined'] * 0.25 + 
             df['reversal_prob'] * 0.25 + 
             composite_efficiency * 0.25)
    
    return factor
