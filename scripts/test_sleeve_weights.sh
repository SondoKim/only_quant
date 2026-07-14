#!/bin/bash
# 슬리브 가중치 사전등록 A/B — 각 변형의 SR을 한 줄로 추출
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/.."

run() {
  local label="$1"; local cfg="$2"
  local out
  out=$(python scripts/run_sleeve_backtest.py --start-date 2016-01-01 ${cfg:+--config-json "$cfg"} 2>/dev/null)
  echo "=== $label ==="
  echo "$out" | grep -E "RATES|FX|PORTFOLIO|H1|H2|북스톱|book.stop" | head -8
  echo
}

run "V0 현행 (t1.0 v0.5 c1.0 p1.0, rev0.5)" ""
run "V1 균등 (value 1.0)" '{"sleeve_weights":{"trend":1.0,"value":1.0,"carry":1.0,"curve":0.0,"policy":1.0}}'
run "V2 value 제거" '{"sleeve_weights":{"trend":1.0,"value":0.0,"carry":1.0,"curve":0.0,"policy":1.0}}'
run "V3 trend 경량 (0.5)" '{"sleeve_weights":{"trend":0.5,"value":0.5,"carry":1.0,"curve":0.0,"policy":1.0}}'
run "V4 carry 중량 (1.5)" '{"sleeve_weights":{"trend":1.0,"value":0.5,"carry":1.5,"curve":0.0,"policy":1.0}}'
run "V5 reversion 0.3" '{"reversion":{"enabled":true,"weight":0.3,"lookback":10,"z_window":120,"smooth":0.5}}'
run "V6 reversion 0.7" '{"reversion":{"enabled":true,"weight":0.7,"lookback":10,"z_window":120,"smooth":0.5}}'
