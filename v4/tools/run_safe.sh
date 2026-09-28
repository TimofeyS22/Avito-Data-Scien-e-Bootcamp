#!/bin/bash
# Запуск тяжёлой команды под «сторожем» памяти: команда работает с пониженным приоритетом,
# а сторож каждые 3 секунды проверяет своп и свободную память и немедленно завершает
# процесс, если система начинает задыхаться (так компьютер не зависает).
# usage: run_safe.sh LOGFILE command args...
LOG="$1"; shift
MAXSWAP_MB=${MAXSWAP_MB:-3000}   # допустимый рост свопа, МБ
MINFREE_PCT=${MINFREE_PCT:-10}   # минимальный процент свободной памяти системы
MAXRSS_MB=${MAXRSS_MB:-5000}     # предельная резидентная память самого процесса, МБ
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.6}
export PYTORCH_MPS_LOW_WATERMARK_RATIO=${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.5}
export OPENBLAS_NUM_THREADS=1
swap_now() { sysctl -n vm.swapusage | awk '{for(i=1;i<=NF;i++) if($i=="used") {v=$(i+2); sub("M","",v); print int(v)}}'; }
BASE=$(swap_now)
nice -n 10 "$@" >> "$LOG" 2>&1 &
PID=$!
echo "$(date +%T) started pid $PID: $*" >> "$LOG"
while kill -0 $PID 2>/dev/null; do
  SW=$(swap_now); FREE=$(memory_pressure 2>/dev/null | awk '/free percentage/ {gsub("%","",$5); print $5}')
  RSS=$(( $(ps -o rss= -p $PID 2>/dev/null || echo 0) / 1024 ))
  if [ $((SW - BASE)) -gt $MAXSWAP_MB ] || [ "${FREE:-100}" -lt $MINFREE_PCT ] || [ $RSS -gt $MAXRSS_MB ]; then
    kill -9 $PID; echo "$(date +%T) WATCHDOG: killed pid $PID (swap +$((SW - BASE))MB, free ${FREE}%, rss ${RSS}MB)" >> "$LOG"; exit 3
  fi
  [ $((SECONDS % 30)) -lt 3 ] && echo "$(date +%T) mon: swap +$((SW - BASE))MB free ${FREE}% rss ${RSS}MB" >> "$LOG.mon"
  sleep 3
done
wait $PID; RC=$?; echo "$(date +%T) finished rc=$RC" >> "$LOG"; exit $RC
