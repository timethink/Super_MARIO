#!/bin/bash

# 配置文件路径和QAF文件路径
CONFIG_FILE="configs/mcts_sft.yaml"
QAF_FILE="../MARIO_EVAL/data/math_test.json"

# 定义参数组合
BATCH_SIZES=(4 8 16 32)
N_GENERATE_SAMPLES=(8 16 32)
QUESTION_RANGES=(16 64 128)
ITERATIONS=40

# 循环生成不同的参数组合并执行
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
  for N_GENERATE_SAMPLE in "${N_GENERATE_SAMPLES[@]}"; do
    for QUESTION_RANGE in "${QUESTION_RANGES[@]}"; do
      # 修改配置文件
      echo "Updating YAML config: batch_size=$BATCH_SIZE, n_generate_sample=$N_GENERATE_SAMPLE, question_range=$QUESTION_RANGE"
      
      # 使用 yq 修改 YAML 配置文件
      yq eval ".batch_size = $BATCH_SIZE" -i "$CONFIG_FILE"
      yq eval ".n_generate_sample = $N_GENERATE_SAMPLE" -i "$CONFIG_FILE"
      yq eval ".best_of = $N_GENERATE_SAMPLE" -i "$CONFIG_FILE"
      yq eval ".iterations = $ITERATIONS" -i "$CONFIG_FILE"
      yq eval ".question_range = $QUESTION_RANGE" -i "$CONFIG_FILE"

      # 执行 solver_demo.py
      echo "Running solver_demo.py with updated config..."
      python solver_demo.py --custom_cfg "$CONFIG_FILE" --qaf "$QAF_FILE"

      if [[ $? -eq 0 ]]; then
        echo "Solver ran successfully."
      else
        echo "Error occurred while running the solver."
      fi
    done
  done
done
