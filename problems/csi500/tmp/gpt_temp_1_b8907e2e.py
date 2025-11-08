import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Amount-Driven Order Flow Imbalance Alpha Factor
    Analyzes directional amount patterns and their relationship to price impact
    """
    
    # Calculate daily returns and price changes
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['price_change'] = df['close'] - df['close'].shift(1)
    df['abs_price_change'] = abs(df['price_change'])
    
    # 1. Analyze Directional Amount Patterns
    # Calculate net order flow (amount on up days vs down days)
    df['up_day_amount'] = np.where(df['returns'] > 0, df['amount'], 0)
    df['down_day_amount'] = np.where(df['returns'] < 0, df['amount'], 0)
    
    # Cumulative directional amount (5-day rolling)
    df['cum_up_amount_5d'] = df['up_day_amount'].rolling(window=5).sum()
    df['cum_down_amount_5d'] = df['down_day_amount'].rolling(window=5).sum()
    df['net_order_flow_5d'] = (df['cum_up_amount_5d'] - df['cum_down_amount_5d']) / (df['cum_up_amount_5d'] + df['cum_down_amount_5d'])
    
    # Detect order flow momentum (consecutive same-direction flow)
    df['flow_direction'] = np.sign(df['net_order_flow_5d'])
    df['flow_momentum'] = df['flow_direction'].rolling(window=3).apply(
        lambda x: len(x[x == x.iloc[-1]]) if len(x) == 3 else np.nan, raw=False
    )
    
    # 2. Relate Order Flow to Price Impact
    # Compute flow efficiency (price change per unit amount)
    df['flow_efficiency'] = df['abs_price_change'] / df['amount']
    
    # Compare to historical efficiency (20-day median)
    df['efficiency_ratio'] = df['flow_efficiency'] / df['flow_efficiency'].rolling(window=20).median()
    
    # Identify inefficient flow patterns
    df['high_amount_low_impact'] = (df['amount'] > df['amount'].rolling(window=20).median() * 1.5) & \
                                  (df['abs_price_change'] < df['abs_price_change'].rolling(window=20).median() * 0.8)
    
    df['low_amount_high_impact'] = (df['amount'] < df['amount'].rolling(window=20).median() * 0.8) & \
                                  (df['abs_price_change'] > df['abs_price_change'].rolling(window=20).median() * 1.5)
    
    # Flow-price divergence indicator
    df['flow_price_divergence'] = (df['net_order_flow_5d'] * np.sign(df['returns'])).rolling(window=5).mean()
    
    # 3. Generate Flow-Based Alpha
    # Base signal: flow momentum adjusted by efficiency
    df['base_signal'] = df['net_order_flow_5d'] * df['efficiency_ratio'] * df['flow_momentum']
    
    # Adjust for inefficient flow patterns
    df['inefficiency_adjustment'] = 1.0
    df.loc[df['high_amount_low_impact'], 'inefficiency_adjustment'] = -0.5  # Expect reversal
    df.loc[df['low_amount_high_impact'], 'inefficiency_adjustment'] = 1.5   # Expect momentum
    
    # Apply divergence adjustment
    df['divergence_adjustment'] = np.where(
        df['flow_price_divergence'] < -0.1,  # Strong negative divergence
        1.2,  # Expect stronger momentum
        np.where(df['flow_price_divergence'] > 0.1,  # Strong positive divergence
                0.8,  # Expect weaker momentum or reversal
                1.0)  # No adjustment
    )
    
    # Combine all components
    df['alpha_factor'] = df['base_signal'] * df['inefficiency_adjustment'] * df['divergence_adjustment']
    
    # Smooth the final factor with 3-day EMA
    df['alpha_factor_smooth'] = df['alpha_factor'].ewm(span=3).mean()
    
    # Remove extreme outliers (winsorize at 5th and 95th percentiles)
    lower_bound = df['alpha_factor_smooth'].quantile(0.05)
    upper_bound = df['alpha_factor_smooth'].quantile(0.95)
    df['alpha_factor_final'] = np.clip(df['alpha_factor_smooth'], lower_bound, upper_bound)
    
    # Normalize by recent volatility (10-day rolling std of returns)
    vol_adj = df['returns'].rolling(window=10).std()
    df['alpha_factor_vol_adj'] = df['alpha_factor_final'] / (vol_adj + 1e-8)
    
    return df['alpha_factor_vol_adj']
