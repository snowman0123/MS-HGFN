import torch


def daily_simulate_trading(prices, buy_signals):
    cash = 10000  # 初始资金
    stocks = 0  # 初始股票数量
    total_assets = []  # 存储每天的总资产

    for i in range(len(prices)):
        if stocks > 0:
            # 如果持有股票，在第二天卖出
            cash += stocks * prices[i]
            stocks = 0

        # 检查是否应该在当前日期买入股票
        if i < len(prices) - 1 and buy_signals[i + 1] == 1:
            stocks = cash // prices[i]
            cash -= stocks * prices[i]

        # 计算当天的总资产
        total_assets.append(cash + stocks * prices[i])

    return total_assets


# 示例价格和买入信号
prices = torch.tensor     ([10, 15, 14, 13, 12, 15, 12, 11, 15, 10])
buy_signals = torch.tensor([1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1])  # 1表示下一天的买入信号

# 模拟交易并打印结果
total_assets = daily_simulate_trading(prices, buy_signals)
total_assets_list = total_assets  # 将结果转换为列表
print(total_assets_list)
