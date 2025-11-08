#!/bin/bash

# Ground-A-Video 批量運行腳本
# 簡單好用版本

# 配置
CONDA_ENV="groundvideo_rtx5090"
CLIP_LENGTH_DEFAULT=15
CLIP_LENGTH_OVERRIDE_CONFIG="p2p/n_u_c_s"
CLIP_LENGTH_OVERRIDE_VALUE=12
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 顏色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 創建日誌目錄
mkdir -p logs
LOG_FILE="logs/run_$(date +%Y%m%d_%H%M%S).log"

echo "========================================" | tee "$LOG_FILE"
echo "Ground-A-Video 批量運行" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "開始時間: $(date)" | tee -a "$LOG_FILE"
echo "Conda 環境: $CONDA_ENV" | tee -a "$LOG_FILE"
echo "Default clip length: $CLIP_LENGTH_DEFAULT" | tee -a "$LOG_FILE"
echo "Override: $CLIP_LENGTH_OVERRIDE_CONFIG -> $CLIP_LENGTH_OVERRIDE_VALUE" | tee -a "$LOG_FILE"
echo "日誌文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# 檢查 conda 環境
if ! conda env list | grep -q "^$CONDA_ENV "; then
    echo -e "${RED}錯誤: Conda 環境 '$CONDA_ENV' 不存在${NC}" | tee -a "$LOG_FILE"
    exit 1
fi

# 計數器
total=0
success=0
failed=0
failed_list=()

# 運行單個配置
run_config() {
    local config=$1
    local category=$2
    local name=$(basename "$config" .yaml)

    # 決定 clip_length：僅對特定配置覆蓋
    local clip_length=$CLIP_LENGTH_DEFAULT
    local current_key="$category/$name"
    if [ "$current_key" == "$CLIP_LENGTH_OVERRIDE_CONFIG" ]; then
        clip_length=$CLIP_LENGTH_OVERRIDE_VALUE
    fi

    echo "" | tee -a "$LOG_FILE"
    echo -e "${YELLOW}[$(date +%H:%M:%S)] 正在運行: $category/$name (clip_length=$clip_length)${NC}" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"

    ((total++))

    # 使用 conda run 來運行，確保在正確的環境中
    if conda run -n "$CONDA_ENV" python main.py \
        --config "video_configs/$category/$name.yaml" \
        --folder "outputs/$category/$name" \
        --clip_length "$clip_length" 2>&1 | tee -a "$LOG_FILE"; then
        echo -e "${GREEN}✓ [$(date +%H:%M:%S)] 成功: $category/$name${NC}" | tee -a "$LOG_FILE"
        ((success++))
        return 0
    else
        echo -e "${RED}✗ [$(date +%H:%M:%S)] 失敗: $category/$name${NC}" | tee -a "$LOG_FILE"
        ((failed++))
        failed_list+=("$category/$name")
        return 1
    fi
}

# 運行所有配置
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "開始處理配置文件..." | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# O2O
if [ -d "video_configs/o2o" ]; then
    echo "" | tee -a "$LOG_FILE"
    echo ">>> 處理 O2O 配置 <<<" | tee -a "$LOG_FILE"
    for yaml in video_configs/o2o/*.yaml; do
        [ -f "$yaml" ] && run_config "$yaml" "o2o"
    done
fi

# O2P
if [ -d "video_configs/o2p" ]; then
    echo "" | tee -a "$LOG_FILE"
    echo ">>> 處理 O2P 配置 <<<" | tee -a "$LOG_FILE"
    for yaml in video_configs/o2p/*.yaml; do
        [ -f "$yaml" ] && run_config "$yaml" "o2p"
    done
fi

# P2P
if [ -d "video_configs/p2p" ]; then
    echo "" | tee -a "$LOG_FILE"
    echo ">>> 處理 P2P 配置 <<<" | tee -a "$LOG_FILE"
    for yaml in video_configs/p2p/*.yaml; do
        [ -f "$yaml" ] && run_config "$yaml" "p2p"
    done
fi

# 總結
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "總結" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "結束時間: $(date)" | tee -a "$LOG_FILE"
echo "總共:     $total" | tee -a "$LOG_FILE"
echo -e "${GREEN}成功:     $success${NC}" | tee -a "$LOG_FILE"
echo -e "${RED}失敗:     $failed${NC}" | tee -a "$LOG_FILE"

if [ $failed -gt 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "失敗的配置:" | tee -a "$LOG_FILE"
    for item in "${failed_list[@]}"; do
        echo "  ✗ $item" | tee -a "$LOG_FILE"
    done
fi

echo "" | tee -a "$LOG_FILE"
echo "日誌已保存到: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

exit $failed
