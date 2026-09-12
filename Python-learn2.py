def count_chars(text):
    """统计字符串中每个字符出现的次数，返回一个 dict"""
    counts = {}                          # 空字典
    for ch in text:                      # 逐字符遍历
        counts[ch] = counts.get(ch, 0) + 1
    return counts

result = count_chars("abracadabra")
print(result)