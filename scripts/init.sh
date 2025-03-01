#!/bin/bash

# 创建 hooks 文件,并添加 git pre-commit 钩子,当提交时执行 scripts/generate_requirements.sh
mkdir -p .git/hooks
cp scripts/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

