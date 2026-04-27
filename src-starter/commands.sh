#!/usr/bin/env bash
set -e

echo "=== Compiling and running all test cases ==="

# baseline: N=2048, d=32, K=16, P=4, rows_per_pe=128
cslc --arch=wse2 layout.csl --fabric-dims=11,6 --fabric-offsets=4,1 \
  --params=P:4,d_dim:32,rows_per_pe:128,K:16 --memcpy --channels=1 -o out_baseline
cs_python run.py --name out_baseline --case baseline

# k=1: N=1024, d=32, K=1, P=2, rows_per_pe=256
cslc --arch=wse2 layout.csl --fabric-dims=11,6 --fabric-offsets=4,1 \
  --params=P:2,d_dim:32,rows_per_pe:256,K:1 --memcpy --channels=1 -o out_k1
cs_python run.py --name out_k1 --case k=1

# k=256: N=1024, d=16, K=256, P=2, rows_per_pe=256
cslc --arch=wse2 layout.csl --fabric-dims=11,6 --fabric-offsets=4,1 \
  --params=P:2,d_dim:16,rows_per_pe:256,K:256 --memcpy --channels=1 -o out_k256
cs_python run.py --name out_k256 --case k=256

# uneven: N=1009, d=32, K=16, P=4, rows_per_pe=64 (ceil(1009/16)=64)
cslc --arch=wse2 layout.csl --fabric-dims=11,6 --fabric-offsets=4,1 \
  --params=P:4,d_dim:32,rows_per_pe:64,K:16 --memcpy --channels=1 -o out_uneven
cs_python run.py --name out_uneven --case uneven

# all_equal: same params as baseline
cslc --arch=wse2 layout.csl --fabric-dims=11,6 --fabric-offsets=4,1 \
  --params=P:4,d_dim:32,rows_per_pe:128,K:16 --memcpy --channels=1 -o out_allequal
cs_python run.py --name out_allequal --case all_equal

# duplicates: same params as baseline  
cslc --arch=wse2 layout.csl --fabric-dims=11,6 --fabric-offsets=4,1 \
  --params=P:4,d_dim:32,rows_per_pe:128,K:16 --memcpy --channels=1 -o out_duplicates
cs_python run.py --name out_duplicates --case duplicates

echo "=== All cases passed ==="
