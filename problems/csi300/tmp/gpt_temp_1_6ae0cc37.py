import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Price Momentum
    ma14 = df['close'].rolling(window=14).mean()
    ma50 = df['close'].rolling(window=50).mean()
    momentum_score = (ma14 > ma50).astype(int) * 2 - 1
    
    # Volume Increase on Price Rise
    volume_increase_score = pd.Series(0, index=df.index)
    price_rise = (df['close'] > df['close'].shift(1))
    volume_increase = (df['volume'] > df['volume'].shift(1)) & (df['volume'] / df['volume'].shift(1) > 1.3)
    volume_increase_score[price_rise & volume_increase] = 1
    
    # Price Volatility
    daily_range = df['high'] - df['low']
    avg_daily_range_20 = daily_range.rolling(window=20).mean()
    volatility_factor = (daily_range > avg_daily_range_20).astype(int) * 2 - 1
    
    # Money Flow Index (MFI) Based Factor
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    
    positive_money_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
    negative_money_flow = -money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
    
    mfi = 100 - (100 / (1 + (positive_money_flow / negative_money_flow)))
    mfi_score = 0
    mfi_score[(mfi < 20)] = 1
    mfi_score[(mfi > 80)] = -1
    
    # Combine all factors into a single alpha factor
    alpha_factor = momentum_score + volume_increase_score + volatility_factor + mfi_score
    return alpha_factor
