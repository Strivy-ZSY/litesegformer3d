import re

def format_params(param_str):
    """
    将形如 'Trainable params: 1,385,016' 的字符串转换为 'Trainable params: 1.39M'
    """
    match = re.search(r'[\d,]+', param_str)
    if not match:
        return param_str  # 如果没有数字，原样返回
    num_str = match.group().replace(',', '')  # 移除逗号
    num = int(num_str)
    num_m = num / 1_000_000  # 转换为 M
    # 保留两位小数，四舍五入，并添加 M
    formatted_num = f"{num_m:.2f}M"
    # 将原字符串中的数字部分替换为格式化后的结果
    result = re.sub(r'[\d,]+', formatted_num, param_str, count=1)
    return result
