#!/usr/bin/env bash
# Reproduces every table and figure. Place the datasets under data/ first (see README).
set -e
for p in P1 P2 P3; do for s in 0 1 2; do python run.py $p bin $s; done; done
for f in 0 1 2 3 4; do SAVE_PRED=1 RESULTS_DIR=results_pred python run.py P4 bin $f; cp results_pred/P4_bin_$f.json results/; done
for s in 0 1 2; do python run.py P6 bin $s; done
python run.py P3 multi 0; python run.py P3 multi 1; python run.py P3 multi 2
for f in 0 1 2 3 4; do python run.py P4 multi $f; done
for fam in backdoor ddos dos injection mitm password ransomware scanning xss; do python zeroday.py $fam 0; done
for s in 1 2; do for fam in backdoor ddos dos injection mitm password ransomware scanning xss; do FAST=1 RESULTS_DIR=results_zdseeds python zeroday.py $fam $s; done; done
for f in 0 1 2 3 4; do DROP_FEATURES=dns_query RESULTS_DIR=results_nodns python run.py P4 bin $f --models RF,LGBM,XGB_all,XGB_MI; done
for fam in backdoor ddos dos injection mitm password ransomware scanning xss; do DROP_FEATURES=dns_query NO_CNN=1 RESULTS_DIR=results_nodns python zeroday.py $fam 0; done
for f in 0 1 2 3 4; do PERM_SEED=7 RESULTS_DIR=results_perm python run.py P4 bin $f --models CNN; done
# Edge-IIoTset (15 classes)
mkdir -p edge
STRICT=tcp.checksum,icmp.checksum,tcp.ack_raw,udp.stream
for s in 0 1 2; do DATASET=edge RESULTS_DIR=edge/res_std python run.py P1 multi $s; done
for s in 0 1 2; do DATASET=edge DROP_FEATURES=$STRICT MIN_CLASS_N=20 RESULTS_DIR=edge/res_strict python run.py P3 multi $s; done
for fam in backdoor ddos_http ddos_icmp ddos_tcp ddos_udp mitm port_scanning ransomware sql_injection uploading vulnerability_scanner xss; do
  DATASET=edge DROP_FEATURES=$STRICT MIN_CLASS_N=20 RESULTS_DIR=edge/res_strict python zeroday.py $fam 0; done
python aggregate.py && python agg2.py && python scripts/host_bootstrap.py
# Host-aware refinements (Section 5.10)
V2=XGB_thr,XGB_cs,CF_H,CF_H_thr,CF_H_cs,CF_v2
for f in 0 1 2 3 4; do RESULTS_DIR=results_v2 python run.py P4 bin $f --models $V2; done
for s in 0 1 2; do DATASET=edge DROP_FEATURES=$STRICT MIN_CLASS_N=20 RESULTS_DIR=edge/res_v2 python run.py P3 multi $s --models XGB_MI,XGB_cs; done
for s in 0 1 2; do for fam in backdoor ddos dos injection mitm password ransomware scanning xss; do CS_ONLY=1 RESULTS_DIR=results_v2zd python zeroday.py $fam $s; done; done
# Reviewer-oriented controls: MLP front-end, host sensitivity, mixed-effects statistics
for f in 0 1 2 3 4; do SAVE_PRED=1 RESULTS_DIR=results_mlp python run.py P4 bin $f --models CF_MLP,CF_MLP_MI; done
for s in 0 1 2; do RESULTS_DIR=results_mlp python run.py P3 bin $s --models CF_MLP,CF_MLP_MI; done
python scripts/host_sensitivity.py && python scripts/frontend_bootstrap.py && python scripts/mixed_effects.py
