// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#define CONSAN_TARGET "amdgcn-amd-amdhsa--gfx1250"
#define CONSAN_METADATA_TARGET CONSAN_TARGET
#define CONSAN_KERNEL lds_producer_consumer
#define CONSAN_WAVE_SIZE 32
#define CONSAN_LEGACY 0
#define GROUP_BYTES 1024
#define CONSAN_MAX_WORKGROUP 1024
#ifndef CONSAN_WIDTH
#define CONSAN_WIDTH 16
#endif
#ifndef CONSAN_SIGNED
#define CONSAN_SIGNED 0
#endif
#ifndef CONSAN_HIGH
#define CONSAN_HIGH 0
#endif
#ifndef CONSAN_ALIAS
#define CONSAN_ALIAS 0
#endif
#ifndef CONSAN_COLLISION
#define CONSAN_COLLISION 0
#endif

.amdgcn_target CONSAN_TARGET
.text
.globl CONSAN_KERNEL
.type CONSAN_KERNEL,@function
.p2align 8
CONSAN_KERNEL:
  s_load_b32 s4, s[0:1], 0
  s_load_b64 s[6:7], s[0:1], 8
  s_wait_kmcnt 0
  v_lshrrev_b32 v6, 5, v0
  v_lshlrev_b32 v1, 4, v6
  v_lshlrev_b32 v7, 4, v6
  v_add_nc_u32 v2, s4, v6
#if CONSAN_WIDTH == 16
  v_add_nc_u32 v2, 0x8000, v2
#endif
  v_mov_b32 v3, 0x89abcdef
  v_mov_b32 v4, 0x13579bdf
  v_mov_b32 v5, 0xfedcba98
  v_mov_b32 v8, 0xdeadbeef
  v_mov_b32 v9, 0x01234567
  v_mov_b32 v10, 0x76543210
  v_mov_b32 v11, 0xa5a5a5a5
  v_and_b32 v12, 31, v0
#if CONSAN_COLLISION
  v_cmp_gt_u32 vcc_lo, 2, v12
#else
  v_cmp_eq_u32 vcc_lo, (31*CONSAN_HIGH), v12
#endif
  s_and_saveexec_b32 s8, vcc_lo
#if CONSAN_WIDTH == 8
  ds_store_b8 v1, v2
#elif CONSAN_WIDTH == 96
  ds_store_b96 v1, v[2:4]
#elif CONSAN_WIDTH == 17
  ds_store_b16 v1, v2
  s_wait_dscnt 0
  ds_load_u16_d16_hi v8, v1
#elif CONSAN_WIDTH == 16
  ds_store_b16 v1, v2 offset:2
  s_wait_dscnt 0
#if CONSAN_SIGNED
  ds_load_i16 v8, v1 offset:2
#else
  ds_load_u16 v8, v1 offset:2
#endif
#elif CONSAN_WIDTH == 64
  ds_store_b64 v1, v[2:3] offset:8
  s_wait_dscnt 0
  ds_load_b64 v[8:9], v1 offset:8
#elif CONSAN_WIDTH == 128
  ds_store_b128 v1, v[2:5] offset:16
  s_wait_dscnt 0
#if CONSAN_ALIAS
  v_mov_b32 v8, v1
  ds_load_b128 v[8:11], v8 offset:16
#else
  ds_load_b128 v[8:11], v1 offset:16
#endif
#endif
  s_wait_dscnt 0
  global_store_b32 v7, v8, s[6:7]
  v_add_nc_u32 v7, 4, v7
  global_store_b32 v7, v9, s[6:7]
  v_add_nc_u32 v7, 4, v7
  global_store_b32 v7, v10, s[6:7]
  v_add_nc_u32 v7, 4, v7
  global_store_b32 v7, v11, s[6:7]
  s_mov_b32 exec_lo, s8
  s_wait_storecnt 0
  s_endpgm
.Lkernel_end:
.size CONSAN_KERNEL, .Lkernel_end-CONSAN_KERNEL

.rodata
.p2align 6
.amdhsa_kernel CONSAN_KERNEL
  .amdhsa_group_segment_fixed_size GROUP_BYTES
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 16
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  // gfx1100 wave32 requires at least 16 user+system SGPR inputs. The
  // kernarg pointer stays at s[0:1]; padding moves workgroup X to s15.
  .amdhsa_user_sgpr_count 15
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr 16
  .amdhsa_next_free_sgpr 16
#if CONSAN_LEGACY
  .amdhsa_accum_offset 8
#endif
#if !CONSAN_LEGACY
#if CONSAN_WAVE_SIZE == 32
  .amdhsa_wavefront_size32 1
#else
  .amdhsa_wavefront_size32 0
#endif
#endif
  .amdhsa_reserve_vcc 1
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.target: CONSAN_METADATA_TARGET
amdhsa.kernels:
  - .name: CONSAN_KERNEL
    .symbol: CONSAN_KERNEL.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: GROUP_BYTES
    .private_segment_fixed_size: 0
    .sgpr_count: 16
    .vgpr_count: 16
    .wavefront_size: CONSAN_WAVE_SIZE
    .max_flat_workgroup_size: CONSAN_MAX_WORKGROUP
    .args:
      - .name: input
        .size: 4
        .offset: 0
        .value_kind: by_value
      - .name: output
        .size: 8
        .offset: 8
        .value_kind: global_buffer
        .address_space: global
...
.end_amdgpu_metadata
