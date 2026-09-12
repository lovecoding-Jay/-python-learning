def mean(nums):
    """返回列表中所有数的平均值（不使用 sum）"""
    if len(nums) == 0:
        raise ValueError("列表不能为空")
    total = 0                       # 累加器，从 0 开始
    for x in nums:                  # 依次取出每个数
        total += x                  # total = total + x
    return total / len(nums)        # 除以个数得到平均值

print(mean([85, 90, 78, 95]))