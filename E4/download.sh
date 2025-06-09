#!/bin/bash

# 设置数据集的URL
DATASET_URL="https://www.depeca.uah.es/colonoscopy_dataset/"

# 设置下载目标目录 (当前目录)
DOWNLOAD_DIR="./"

echo "正在尝试从 ${DATASET_URL} 下载核心数据集文件到 ${DOWNLOAD_DIR}"
echo "将下载 .mp4, .xml, .txt, .xlsx 文件..."

# 使用 wget 命令进行递归下载
# 加上 --no-check-certificate 来跳过证书验证
# -r: 递归下载
# -np: 不下载父目录
# -nH: 不创建以主机名命名的顶级目录
# --cut-dirs=0: 从远程目录路径中剪切0个组件，将内容直接放在当前目录下的相应结构中。
# -A "*.mp4,*.xml,*.txt,*.xlsx": 只接受指定的核心文件类型进行下载
# -c: 继续下载已部分下载的文件 (如果下载中断可以恢复)
wget --no-check-certificate -r -np -nH --cut-dirs=0 -A "*.mp4,*.xml,*.txt,*.xlsx" -P "${DOWNLOAD_DIR}" "${DATASET_URL}"

# 检查wget命令的退出状态
if [ $? -eq 0 ]; then
    echo "核心数据集文件下载完成！请检查当前目录下的文件和文件夹。"
    echo "主要数据应位于 ./Colonoscopy_Video_Dataset/ 文件夹下，标签和信息文件应在当前目录。"
else
    echo "核心数据集文件下载过程中可能发生错误。请检查网络连接或重试。"
fi