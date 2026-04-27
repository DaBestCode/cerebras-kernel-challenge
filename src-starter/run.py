#!/usr/bin/env cs_python

import argparse
import json
import numpy as np
import sys

from reference import (topk_reference, make_baseline, make_k_eq_1,
                        make_k_large, make_uneven, make_all_equal, make_duplicates)

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime      # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType  # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder     # pylint: disable=no-name-in-module

parser = argparse.ArgumentParser()
parser.add_argument("--name",   help="path to compiled binary folder")
parser.add_argument("--cmaddr", help="IP:port for real CS system")
parser.add_argument("--case",   default="baseline", help="test case name")
args = parser.parse_args()

with open(f"{args.name}/out.json", encoding='utf-8') as f:
    compile_data = json.load(f)

P           = int(compile_data['params']['P'])
d_dim       = int(compile_data['params']['d_dim'])
rows_per_pe = int(compile_data['params']['rows_per_pe'])
K           = int(compile_data['params']['K'])

print(f"Loaded params: P={P}, d_dim={d_dim}, rows_per_pe={rows_per_pe}, K={K}")

cases = {
    "baseline":   make_baseline,
    "k=1":        make_k_eq_1,
    "k=256":      make_k_large,
    "uneven":     make_uneven,
    "all_equal":  make_all_equal,
    "duplicates": make_duplicates,
}
case = cases[args.case]()
D = case["D"]
q = case["q"]
N = D.shape[0]

print(f"Test case '{args.case}': N={N}, d={D.shape[1]}, K={K}, P={P}")

# Pad D to P*P*rows_per_pe rows — padding rows are 1e19 so distance >> any real row
total_rows = P * P * rows_per_pe
D_padded   = np.full((total_rows, d_dim), 1e19, dtype=np.float32)
D_padded[:N] = D

# Reshape for ROW_MAJOR memcpy: (Ph, Pw, rows_per_pe*d_dim)
# PE at grid position (pe_x, pe_y) receives D_flat[pe_y, pe_x, :]
D_flat = D_padded.reshape(P, P, rows_per_pe * d_dim)

# Tile q for broadcast: every PE gets a copy
q_tiled = np.tile(q, (P, P, 1)).astype(np.float32)

# Reference answer
ref_indices, ref_distances = topk_reference(D, q, K)
print(f"Reference top-{K} indices:   {ref_indices}")
print(f"Reference top-{K} distances: {ref_distances}")

runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
memcpy_order = MemcpyOrder.ROW_MAJOR

symbol_D         = runner.get_id("D")
symbol_q         = runner.get_id("q")
symbol_indices   = runner.get_id("indices")
symbol_distances = runner.get_id("distances")

runner.load()
runner.run()

print("Copying D to PEs...")
runner.memcpy_h2d(symbol_D, D_flat.ravel(), 0, 0, P, P, rows_per_pe * d_dim,
                  streaming=False, data_type=memcpy_dtype,
                  nonblock=False, order=memcpy_order)

print("Broadcasting q to all PEs...")
runner.memcpy_h2d(symbol_q, q_tiled.ravel(), 0, 0, P, P, d_dim,
                  streaming=False, data_type=memcpy_dtype,
                  nonblock=False, order=memcpy_order)

print("Launching kernel...")
runner.launch("main", nonblock=False)

# Result lives at PE(pe_x=0, pe_y=P-1) — bottom-left of the grid
out_indices   = np.zeros(K, dtype=np.int32)
out_distances = np.zeros(K, dtype=np.float32)

runner.memcpy_d2h(out_indices, symbol_indices, 0, P-1, 1, 1, K,
                  streaming=False, data_type=memcpy_dtype,
                  nonblock=False, order=memcpy_order)
runner.memcpy_d2h(out_distances, symbol_distances, 0, P-1, 1, 1, K,
                  streaming=False, data_type=memcpy_dtype,
                  nonblock=False, order=memcpy_order)
runner.stop()

print(f"Kernel indices:   {out_indices}")
print(f"Kernel distances: {out_distances}")

indices_match   = np.array_equal(out_indices, ref_indices)
distances_match = np.allclose(out_distances, ref_distances, atol=1e-3, rtol=0)

if indices_match and distances_match:
    print(f"\n✅ PASS: {args.case}")
else:
    if not indices_match:
        print(f"\n❌ FAIL indices:")
        print(f"   got:      {out_indices}")
        print(f"   expected: {ref_indices}")
    if not distances_match:
        print(f"\n❌ FAIL distances:")
        print(f"   got:      {out_distances}")
        print(f"   expected: {ref_distances}")
    sys.exit(1)
