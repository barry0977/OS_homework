#!/bin/bash
MNT="/home/czx/kvm/project4/gptFS"
mkdir -p "$MNT"
"$(which python)" gptfs.py "$MNT" &   # 后台挂载
sleep 1

SESSION="$MNT/session1"
mkdir "$SESSION"
echo "你好gpt" > "$SESSION/input"
sleep 5

echo "[输出]:"
cat "$SESSION/output"
echo "[错误]:"
cat "$SESSION/error"

umount "$MNT"
