#!/bin/bash
# 片足立ちモデルを歩行モデルに修正するワンコマンドスクリプト

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  片足立ちモデル修正スクリプト - 3段階の自動最適化                 ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")" || exit 1

# ステップ 1: 強制前進モード (3ループ)
echo "🚀 STEP 1/3: Force forward mode (3 loops)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 hrp2_auto_train_loop.py \
    -e hrp2-walking-fixed-v1 \
    --max_iterations 1000 \
    --num_loops 3 \
    --force_forward

if [ $? -ne 0 ]; then
    echo "❌ Step 1 failed"
    exit 1
fi

echo ""
echo "✅ Step 1 completed!"
echo ""

# ステップ 2: 通常モードで精密化 (5ループ)
echo "🚀 STEP 2/3: Normal mode with automatic adjustment (5 loops)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 hrp2_auto_train_loop.py \
    -e hrp2-walking-fixed-v2 \
    --max_iterations 1000 \
    --num_loops 5

if [ $? -ne 0 ]; then
    echo "❌ Step 2 failed"
    exit 1
fi

echo ""
echo "✅ Step 2 completed!"
echo ""

# ステップ 3: 最終検証
echo "🚀 STEP 3/3: Final verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 test_walk_score.py -e hrp2-walking-fixed-v2_loop5 --verbose

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  ✅ 修正完了！                                                    ║"
echo "║  最終モデル: logs/hrp2-walking-fixed-v2_loop5/model_500.pt         ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
