# 下载 clash ubuntu
wget https://github.com/Dreamacro/clash/releases/latest/download/clash-linux-amd64.gz
# 解压
gzip -d clash-linux-amd64.gz
# 赋予执行权限
chmod +x clash-linux-amd64
# 创建配置文件
touch config.yaml
# 编辑配置文件, 配置文件从 window 下载
vi config.yaml

# 启动
./clash-linux-amd64 -d ./
# 查看日志
tail -f clash.log