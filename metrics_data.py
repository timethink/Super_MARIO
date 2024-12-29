import re
import pandas as pd

# 提取指标数据
metrics_text = metrics_data.decode("utf-8")

# 解析特定指标（以 vllm:generation_tokens_total 为例）
pattern = r'vllm:generation_tokens_total\{.*\} (\d+\.?\d*) (\d+)'
matches = re.findall(pattern, metrics_text)

# 转换为 DataFrame
df = pd.DataFrame(matches, columns=["value", "timestamp"])
df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
df["value"] = df["value"].astype(float)

print(df)
