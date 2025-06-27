#!/bin/bash
# 资源监控脚本
# 用于定时采集服务器CPU、内存、磁盘、网络等资源使用情况
# 支持日志输出和告警

LOG_DIR="logs"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/server_monitor_$(date +%Y%m%d_%H%M%S).log"

# 采集间隔（秒）
INTERVAL=60

# 告警阈值
CPU_THRESHOLD=80
MEM_THRESHOLD=85
DISK_THRESHOLD=90

function monitor_once() {
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    CPU=$(top -l 1 | grep "CPU usage" | awk '{print $3}' | sed 's/%//')
    MEM=$(vm_stat | grep "Pages active" | awk '{print $3}' | sed 's/\.//')
    MEM_TOTAL=$(sysctl hw.memsize | awk '{print $2}')
    MEM_USED=$(($(vm_stat | grep "Pages active" | awk '{print $3}' | sed 's/\.//') * 4096))
    MEM_PERCENT=$((100 * $MEM_USED / $MEM_TOTAL))
    DISK=$(df -h / | tail -1 | awk '{print $5}' | sed 's/%//')
    NET=$(netstat -ib | grep -e "en0" | awk '{RX+=$7; TX+=$10} END {print RX, TX}')

    echo "$TIMESTAMP CPU: $CPU% MEM: $MEM_PERCENT% DISK: $DISK% NET: $NET" >> $LOG_FILE

    # 告警
    if [ "$CPU" -gt "$CPU_THRESHOLD" ]; then
        echo "$TIMESTAMP [ALERT] CPU使用率超过阈值: $CPU%" >> $LOG_FILE
    fi
    if [ "$MEM_PERCENT" -gt "$MEM_THRESHOLD" ]; then
        echo "$TIMESTAMP [ALERT] 内存使用率超过阈值: $MEM_PERCENT%" >> $LOG_FILE
    fi
    if [ "$DISK" -gt "$DISK_THRESHOLD" ]; then
        echo "$TIMESTAMP [ALERT] 磁盘使用率超过阈值: $DISK%" >> $LOG_FILE
    fi
}

while true; do
    monitor_once
    sleep $INTERVAL
done 