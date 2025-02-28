#!/bin/bash
set -eo pipefail

echo "===== 安全依赖扫描 ====="
cd "$(dirname "$0")/.." || { echo "路径错误！"; exit 1; }

# 安装最新版 pipreqs
pip install --upgrade pipreqs -i https://pypi.tuna.tsinghua.edu.cn/simple

# 构建参数数组
args=("." "--force") # --force 强制覆盖
args+=("--encoding" "utf-8" "--savepath" "requirements.tmp")

# 执行扫描命令
pipreqs "${args[@]}"

# 检查是否生成 requirements.txt
if [ ! -f "requirements.tmp" ]; then
  echo "生成失败！"
  exit 1
fi

# 移动临时文件
mv requirements.tmp requirements.txt

# # 添加阿里云源
# if ! grep -q "^--i" requirements.txt; then
#   sed -i '1i--i https://mirrors.aliyun.com/pytorch-wheels/cu118' requirements.txt
# fi

echo "生成成功！"
cat requirements.txt