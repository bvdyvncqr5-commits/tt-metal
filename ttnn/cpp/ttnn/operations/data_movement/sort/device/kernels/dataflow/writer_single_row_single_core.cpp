// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

#include "sort_dataflow_common.hpp"

/*
To improve performance of both reader and writer kernels the work has been split so that they both prepare input and
save output data.

Reader:
    * Reads input value data from DRAM and writes it to L1 circular buffer.
    * Write processed index data from L1 to DRAM.

Writer:
    * Generates index input data and writes it to L1 circular buffer.
    * Write output values from L1 to DRAM.
*/
void kernel_main() {
    // Runtime args
    const uint32_t value_tensor_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t core_loop_count = get_arg_val<uint32_t>(1);

    // Compile time args
    constexpr uint32_t value_tensor_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t index_tensor_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t Ht = get_compile_time_arg_val(3);
    constexpr uint32_t total_number_of_cores = get_compile_time_arg_val(4);
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(5);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(6);

    // arg 7: is_32_bit_data – true when fp32_dest_acc_en is enabled (float32 input or
    // uint32 index).  Both TILE and ROW_MAJOR paths must generate uint32 index tiles
    // when the sort kernel runs in 32-bit DEST mode.
    constexpr bool is_32_bit_data = get_compile_time_arg_val(7) == 1;
    constexpr bool is_row_major = get_compile_time_arg_val(8) == 1;
    constexpr uint32_t rm_value_output_dfb_index = get_compile_time_arg_val(9);
    constexpr uint32_t W_value_bytes = get_compile_time_arg_val(10);

    constexpr auto value_tensor_args = TensorAccessorArgs<11>();

    // New args appended after TensorAccessorArgs:
    // is_uint16_fp32_mode – set when input dtype is UINT16. In this mode the sort compute
    // kernel writes sorted values as Float32 (into c_4) to avoid bf16 packing corruption.
    // The writer must convert Float32 → UInt16 element-by-element before writing to DRAM.
    constexpr bool is_uint16_fp32_mode =
        get_compile_time_arg_val(value_tensor_args.next_compile_time_args_offset()) == 1;
    // uint16_conv_cb_index – a 1-tile UInt16 CB used as the TILE-path conversion staging buffer.
    constexpr uint32_t uint16_conv_cb_index =
        get_compile_time_arg_val(value_tensor_args.next_compile_time_args_offset() + 1);
    // rm_uint16_output_stage_cb_index – UInt16 RM staging CB for ROW_MAJOR UINT16 output.
    // The writer converts Float32 RM rows from rm_value_output_dfb to UInt16 here,
    // then DMAs W_value_bytes (one UInt16 row) to DRAM.
    constexpr uint32_t rm_uint16_output_stage_cb_index =
        get_compile_time_arg_val(value_tensor_args.next_compile_time_args_offset() + 2);

    constexpr uint32_t one_tile = 1;

    // TensorAccessor handles both interleaved and sharded buffers natively.
    // For TILE layout: one "page" = one tile.
    // For ROW_MAJOR layout: one "page" = one row of W elements (W_value_bytes).
    const auto value_accessor = TensorAccessor(value_tensor_args, value_tensor_buffer_addr);

    Noc noc;
    DataflowBuffer value_tensor_dfb(value_tensor_cb_index);
    DataflowBuffer rm_value_output_dfb(rm_value_output_dfb_index);
    constexpr uint32_t value_tensor_tile_size = get_tile_size(value_tensor_cb_index);
    constexpr uint32_t uint16_tile_size = get_tile_size(uint16_conv_cb_index);

    if constexpr (!is_row_major) {
        for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
            const uint32_t h = core_loop * total_number_of_cores +
                               get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

            // Generate index tiles into index_tensor_cb_index (consumed by compute)
            for (uint32_t w = 0; w < Wt; w++) {
                if (is_32_bit_data) {
                    generate_index_tile<uint32_t>(index_tensor_cb_index, w);
                } else {
                    generate_index_tile<uint16_t>(index_tensor_cb_index, w);
                }
            }

            if constexpr (is_uint16_fp32_mode) {
                // The compute kernel stored sorted values as Float32 in value_tensor_cb (c_4).
                // Convert each Float32 element back to UInt16 using a staging CB, then DMA.
                // This avoids the bf16-truncation bug that pack_tile would cause when packing
                // a fp32 DEST register to a UInt16 circular buffer.
                DataflowBuffer conv_dfb(uint16_conv_cb_index);
                constexpr uint32_t ELEMENTS_PER_TILE = 1024;  // 32×32

                for (uint32_t w = 0; w < Wt; w++) {
                    value_tensor_dfb.wait_front(one_tile);
                    conv_dfb.reserve_back(one_tile);

                    // Float32 source pointer (4 bytes per element)
                    volatile tt_l1_ptr uint32_t* fp32_ptr =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(value_tensor_dfb.get_read_ptr());
                    // UInt16 destination pointer (2 bytes per element)
                    volatile tt_l1_ptr uint16_t* u16_ptr =
                        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(conv_dfb.get_write_ptr());

                    for (uint32_t i = 0; i < ELEMENTS_PER_TILE; i++) {
                        const uint32_t fp32_bits = fp32_ptr[i];
                        float fval;
                        __builtin_memcpy(&fval, &fp32_bits, sizeof(fval));
                        u16_ptr[i] = static_cast<uint16_t>(static_cast<uint32_t>(fval));
                    }

                    // Drain the RISC-V store buffer so all uint16 writes reach L1
                    // before the NoC DMA reads from the same buffer.  Without this
                    // fence the NoC (a separate L1 client with no program-order
                    // guarantee with RISC-V stores) may observe stale or partially
                    // written values.
                    __sync_synchronize();

                    value_tensor_dfb.pop_front(one_tile);
                    conv_dfb.push_back(one_tile);

                    // DMA the converted UInt16 tile to DRAM using the standard accessor API.
                    conv_dfb.wait_front(one_tile);
                    noc.async_write(
                        conv_dfb,
                        value_accessor,
                        uint16_tile_size,
                        {.offset_bytes = 0},
                        {.page_id = h * Wt + w, .offset_bytes = 0});
                    noc.async_write_barrier();
                    conv_dfb.pop_front(one_tile);
                }
            } else {
                // Write sorted value tiles from value_tensor_dfb → DRAM
                for (uint32_t w = 0; w < Wt; w++) {
                    value_tensor_dfb.wait_front(one_tile);
                    noc.async_write(
                        value_tensor_dfb,
                        value_accessor,
                        value_tensor_tile_size,
                        {.offset_bytes = 0},
                        {.page_id = h * Wt + w, .offset_bytes = 0});
                    noc.async_write_barrier();
                    value_tensor_dfb.pop_front(one_tile);
                }
            }
        }
    } else {
        // ------------------------------------------------------------------
        // ROW_MAJOR path
        //
        // The value accessor's page size = W_value_bytes (one RM row).
        //
        // Per loop iteration we handle one tile-row = 32 consecutive rows:
        //   Input:  generate Wt TILE-format index tiles into index_tensor_cb
        //           (compute kernel sorts them alongside the tilized values).
        //   Output: drain 32 untilized value pages from rm_value_output_dfb
        //           → write via noc.async_write → value DRAM buffer.
        // ------------------------------------------------------------------
        constexpr uint32_t TILE_H = 32;  // TILE_HEIGHT

        for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
            const uint32_t h = core_loop * total_number_of_cores +
                               get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

            // Generate Wt index tiles (TILE / integer format) into index_tensor_cb_index.
            // The topk LLK reads indices via LO16 (uint16) or INT32 (uint32) mode, so
            // the index CB must contain raw unsigned integers, not floating-point values.
            for (uint32_t w = 0; w < Wt; w++) {
                if (is_32_bit_data) {
                    generate_index_tile<uint32_t>(index_tensor_cb_index, w);
                } else {
                    generate_index_tile<uint16_t>(index_tensor_cb_index, w);
                }
            }

            // Drain 32 sorted RM value rows from rm_value_output_dfb → DRAM.
            // For UINT16 inputs rm_value_output_dfb holds Float32 rows; convert
            // each row to UInt16 via the staging CB before writing to DRAM.
            const uint32_t row_base = h * TILE_H;
            if constexpr (is_uint16_fp32_mode) {
                DataflowBuffer rm_u16_out_dfb(rm_uint16_output_stage_cb_index);
                constexpr uint32_t W_elements = W_value_bytes / sizeof(uint16_t);
                for (uint32_t row = 0; row < TILE_H; row++) {
                    rm_value_output_dfb.wait_front(one_tile);
                    rm_u16_out_dfb.reserve_back(one_tile);

                    volatile tt_l1_ptr uint32_t* fp32_src =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rm_value_output_dfb.get_read_ptr());
                    volatile tt_l1_ptr uint16_t* u16_dst =
                        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rm_u16_out_dfb.get_write_ptr());

                    for (uint32_t i = 0; i < W_elements; i++) {
                        const uint32_t fp32_bits = fp32_src[i];
                        float fval;
                        __builtin_memcpy(&fval, &fp32_bits, sizeof(fval));
                        u16_dst[i] = static_cast<uint16_t>(static_cast<uint32_t>(fval));
                    }
                    // Drain RISC-V store buffer before NoC reads from the same buffer.
                    __sync_synchronize();

                    rm_value_output_dfb.pop_front(one_tile);
                    rm_u16_out_dfb.push_back(one_tile);

                    rm_u16_out_dfb.wait_front(one_tile);
                    noc.async_write(
                        rm_u16_out_dfb,
                        value_accessor,
                        W_value_bytes,
                        {.offset_bytes = 0},
                        {.page_id = row_base + row, .offset_bytes = 0});
                    noc.async_write_barrier();
                    rm_u16_out_dfb.pop_front(one_tile);
                }
            } else {
                for (uint32_t row = 0; row < TILE_H; row++) {
                    rm_value_output_dfb.wait_front(one_tile);
                    noc.async_write(
                        rm_value_output_dfb,
                        value_accessor,
                        W_value_bytes,
                        {.offset_bytes = 0},
                        {.page_id = row_base + row, .offset_bytes = 0});
                    noc.async_write_barrier();
                    rm_value_output_dfb.pop_front(one_tile);
                }
            }
        }
    }
}
