// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

#include <cstdint>

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
    const uint32_t input_tensor_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t index_tensor_buffer_addr = get_arg_val<uint32_t>(1);
    const uint32_t core_loop_count = get_arg_val<uint32_t>(2);

    // Compile time args
    constexpr uint32_t input_tensor_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t index_tensor_output_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t Ht = get_compile_time_arg_val(3);
    constexpr uint32_t total_number_of_cores = get_compile_time_arg_val(4);
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(5);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(6);
    constexpr bool is_row_major = get_compile_time_arg_val(7) == 1;
    constexpr uint32_t rm_input_dfb_index = get_compile_time_arg_val(8);
    constexpr uint32_t rm_index_output_dfb_index = get_compile_time_arg_val(9);
    constexpr uint32_t W_value_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t W_index_bytes = get_compile_time_arg_val(11);

    constexpr auto input_tensor_args = TensorAccessorArgs<12>();
    constexpr auto index_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();

    // UINT16 input mode: the hardware unpack cannot numerically convert UInt16 → Float32
    // (ISA spec only allows UInt16 → UInt16 destination).  Instead the reader kernel does
    // a software conversion on the RISC-V core: DMA the raw UInt16 tile into a staging CB
    // (c_12) and then emit float(uint16_val) element-by-element into c_0 (Float32 CB).
    // The compute kernel then sees correct Float32 values and sorts them exactly.
    constexpr bool is_uint16_fp32_mode =
        get_compile_time_arg_val(index_tensor_args.next_compile_time_args_offset()) == 1;
    constexpr uint32_t uint16_input_stage_cb_index =
        get_compile_time_arg_val(index_tensor_args.next_compile_time_args_offset() + 1);
    // ROW_MAJOR UINT16 staging CB: the reader DMAs one raw UInt16 RM row here,
    // then converts element-by-element to Float32 and pushes to rm_input_dfb.
    constexpr uint32_t rm_uint16_input_stage_cb_index =
        get_compile_time_arg_val(index_tensor_args.next_compile_time_args_offset() + 2);

    // Input tensor config
    constexpr uint32_t one_tile = 1;

    // TensorAccessors handle both interleaved and sharded buffers natively.
    // For TILE layout: one "page" in the accessor = one tile.
    // For ROW_MAJOR layout: one "page" in the accessor = one row of W elements.
    const auto input_accessor = TensorAccessor(input_tensor_args, input_tensor_buffer_addr);
    const auto index_accessor = TensorAccessor(index_tensor_args, index_tensor_buffer_addr);

    Noc noc;
    DataflowBuffer input_tensor_dfb(input_tensor_cb_index);
    DataflowBuffer index_output_dfb(index_tensor_output_cb_index);
    DataflowBuffer rm_input_dfb(rm_input_dfb_index);
    DataflowBuffer rm_index_output_dfb(rm_index_output_dfb_index);
    constexpr uint32_t input_tensor_tile_size = get_tile_size(input_tensor_cb_index);
    constexpr uint32_t index_tensor_tile_size = get_tile_size(index_tensor_output_cb_index);
    constexpr uint32_t uint16_stage_tile_size = get_tile_size(uint16_input_stage_cb_index);
    constexpr uint32_t ELEMENTS_PER_TILE = 1024;  // 32×32

    if constexpr (!is_row_major) {
        for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
            const uint32_t h = core_loop * total_number_of_cores +
                               get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

            // Read input tiles from DRAM → tile input CB
            if constexpr (is_uint16_fp32_mode) {
                // Step 1: DMA raw UInt16 tile from DRAM into staging CB (c_12, UInt16 format).
                // Step 2: RISC-V software converts each uint16 → float32 exactly and writes
                //         the float32 tile into c_0 (Float32 CB) for the compute kernel.
                DataflowBuffer uint16_stage_dfb(uint16_input_stage_cb_index);
                for (uint32_t w = 0; w < Wt; w++) {
                    uint16_stage_dfb.reserve_back(one_tile);
                    noc.async_read(
                        input_accessor,
                        uint16_stage_dfb,
                        uint16_stage_tile_size,
                        {.page_id = h * Wt + w, .offset_bytes = 0},
                        {.offset_bytes = 0});
                    noc.async_read_barrier();
                    uint16_stage_dfb.push_back(one_tile);

                    uint16_stage_dfb.wait_front(one_tile);
                    input_tensor_dfb.reserve_back(one_tile);

                    volatile tt_l1_ptr uint16_t* src =
                        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(uint16_stage_dfb.get_read_ptr());
                    volatile tt_l1_ptr uint32_t* dst =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(input_tensor_dfb.get_write_ptr());

                    for (uint32_t i = 0; i < ELEMENTS_PER_TILE; i++) {
                        float fval = static_cast<float>(static_cast<uint32_t>(src[i]));
                        uint32_t bits;
                        __builtin_memcpy(&bits, &fval, sizeof(bits));
                        dst[i] = bits;
                    }

                    uint16_stage_dfb.pop_front(one_tile);
                    input_tensor_dfb.push_back(one_tile);
                }
            } else {
                for (uint32_t w = 0; w < Wt; w++) {
                    input_tensor_dfb.reserve_back(one_tile);
                    noc.async_read(
                        input_accessor,
                        input_tensor_dfb,
                        input_tensor_tile_size,
                        {.page_id = h * Wt + w, .offset_bytes = 0},
                        {.offset_bytes = 0});
                    noc.async_read_barrier();
                    input_tensor_dfb.push_back(one_tile);
                }
            }

            // Write sorted index tiles from index output CB → DRAM
            for (uint32_t w = 0; w < Wt; w++) {
                index_output_dfb.wait_front(one_tile);
                noc.async_write(
                    index_output_dfb,
                    index_accessor,
                    index_tensor_tile_size,
                    {.offset_bytes = 0},
                    {.page_id = h * Wt + w, .offset_bytes = 0});
                noc.async_write_barrier();
                index_output_dfb.pop_front(one_tile);
            }
        }
    } else {
        // ------------------------------------------------------------------
        // ROW_MAJOR path
        //
        // The input accessor's page size = W_value_bytes (one RM row).
        // The index accessor's page size = W_index_bytes (one RM index row).
        //
        // For each tile-row (TILE_HEIGHT = 32 consecutive logical rows):
        //   Input:  read 32 pages via noc.async_read → rm_input_dfb
        //           so the compute kernel can tilize them.
        //   Output: drain 32 untilized index pages from rm_index_output_dfb
        //           → write via noc.async_write → index DRAM buffer.
        // ------------------------------------------------------------------
        constexpr uint32_t TILE_H = 32;  // TILE_HEIGHT

        for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
            const uint32_t h = core_loop * total_number_of_cores +
                               get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

            // Base page index for this tile-row group in the RM buffer
            const uint32_t row_base = h * TILE_H;

            // --- Read TILE_H input rows into rm_input_dfb ---
            if constexpr (is_uint16_fp32_mode) {
                // UInt16 ROW_MAJOR path:
                //   Step 1: DMA one raw UInt16 row (W_value_bytes) into the staging CB.
                //   Step 2: Convert each uint16 element → float32 in software and push
                //           the float32 row to rm_input_dfb (Float32 RM CB).
                DataflowBuffer rm_u16_stage_dfb(rm_uint16_input_stage_cb_index);
                constexpr uint32_t W_elements = W_value_bytes / sizeof(uint16_t);
                for (uint32_t row = 0; row < TILE_H; row++) {
                    rm_u16_stage_dfb.reserve_back(one_tile);
                    noc.async_read(
                        input_accessor,
                        rm_u16_stage_dfb,
                        W_value_bytes,
                        {.page_id = row_base + row, .offset_bytes = 0},
                        {.offset_bytes = 0});
                    noc.async_read_barrier();
                    rm_u16_stage_dfb.push_back(one_tile);

                    rm_u16_stage_dfb.wait_front(one_tile);
                    rm_input_dfb.reserve_back(one_tile);

                    volatile tt_l1_ptr uint16_t* src =
                        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rm_u16_stage_dfb.get_read_ptr());
                    volatile tt_l1_ptr uint32_t* dst =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rm_input_dfb.get_write_ptr());

                    for (uint32_t i = 0; i < W_elements; i++) {
                        float fval = static_cast<float>(static_cast<uint32_t>(src[i]));
                        uint32_t bits;
                        __builtin_memcpy(&bits, &fval, sizeof(bits));
                        dst[i] = bits;
                    }
                    // Drain store buffer before the compute kernel reads this row.
                    __sync_synchronize();

                    rm_u16_stage_dfb.pop_front(one_tile);
                    rm_input_dfb.push_back(one_tile);
                }
            } else {
                for (uint32_t row = 0; row < TILE_H; row++) {
                    rm_input_dfb.reserve_back(one_tile);
                    noc.async_read(
                        input_accessor,
                        rm_input_dfb,
                        W_value_bytes,
                        {.page_id = row_base + row, .offset_bytes = 0},
                        {.offset_bytes = 0});
                    noc.async_read_barrier();
                    rm_input_dfb.push_back(one_tile);
                }
            }

            // --- Drain TILE_H untilized index rows from rm_index_output_dfb → DRAM ---
            //
            // Compute kernel pack_untilize'd Wt sorted index tiles into
            // TILE_HEIGHT contiguous RM pages in rm_index_output_dfb.
            // pack_untilize_block writes uint16/uint32 elements in the natural
            // little-endian layout that the host expects, so no byte swap is
            // required here regardless of the index dtype.
            for (uint32_t row = 0; row < TILE_H; row++) {
                rm_index_output_dfb.wait_front(one_tile);
                noc.async_write(
                    rm_index_output_dfb,
                    index_accessor,
                    W_index_bytes,
                    {.offset_bytes = 0},
                    {.page_id = row_base + row, .offset_bytes = 0});
                noc.async_write_barrier();
                rm_index_output_dfb.pop_front(one_tile);
            }
        }
    }
}
