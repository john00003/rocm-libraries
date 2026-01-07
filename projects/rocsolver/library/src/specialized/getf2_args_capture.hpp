/************************************************************************
 * Argument capture utilities for getf2_small_kernel reproducer
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 * *************************************************************************/

#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <hip/hip_runtime.h>

namespace getf2_capture {

// Structure to hold all arguments for one call to getf2_small_kernel
template <typename T, typename I>
struct Getf2CallArgs {
    I m;
    I n;
    I lda;
    I batch_count;
    I offset;
    rocblas_stride shiftA;
    rocblas_stride strideA;
    rocblas_stride shiftP;
    rocblas_stride strideP;
    rocblas_stride stridePI;
    bool pivot;
    
    // Grid/block dimensions
    int grid_x, grid_y, grid_z;
    int block_x, block_y, block_z;
    size_t lmemsize;
    
    // Matrix data for each batch (host copy)
    std::vector<std::vector<T>> A_data;  // [batch][m*lda]
    std::vector<std::vector<I>> ipiv_data; // [batch][n]
    std::vector<std::vector<I>> permut_idx_data; // [batch][permut_size]
    std::vector<I> info_data; // [batch]
};

// Track call count globally
inline int& getCaptureCallCount() {
    static int count = 0;
    return count;
}

// Save captured arguments to file
template <typename T, typename I>
void saveArgsToFile(const Getf2CallArgs<T, I>& args, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing\n";
        return;
    }
    
    // Write scalar arguments
    file << "# Getf2 Small Kernel Arguments\n";
    file << "m=" << args.m << "\n";
    file << "n=" << args.n << "\n";
    file << "lda=" << args.lda << "\n";
    file << "batch_count=" << args.batch_count << "\n";
    file << "offset=" << args.offset << "\n";
    file << "shiftA=" << args.shiftA << "\n";
    file << "strideA=" << args.strideA << "\n";
    file << "shiftP=" << args.shiftP << "\n";
    file << "strideP=" << args.strideP << "\n";
    file << "stridePI=" << args.stridePI << "\n";
    file << "pivot=" << (args.pivot ? 1 : 0) << "\n";
    file << "grid=" << args.grid_x << "," << args.grid_y << "," << args.grid_z << "\n";
    file << "block=" << args.block_x << "," << args.block_y << "," << args.block_z << "\n";
    file << "lmemsize=" << args.lmemsize << "\n";
    
    // Write matrix data for each batch
    for (I b = 0; b < args.batch_count; ++b) {
        file << "# Batch " << b << " Matrix A (row-major, m=" << args.m << " x n=" << args.n << ")\n";
        for (I i = 0; i < args.m; ++i) {
            for (I j = 0; j < args.n; ++j) {
                if (j > 0) file << ", ";
                file << args.A_data[b][i + j * args.lda];
            }
            file << "\n";
        }
        
        file << "# Batch " << b << " ipiv\n";
        for (I j = 0; j < args.n; ++j) {
            if (j > 0) file << ", ";
            file << args.ipiv_data[b][j];
        }
        file << "\n";
        
        if (!args.permut_idx_data.empty() && !args.permut_idx_data[b].empty()) {
            file << "# Batch " << b << " permut_idx\n";
            for (size_t j = 0; j < args.permut_idx_data[b].size(); ++j) {
                if (j > 0) file << ", ";
                file << args.permut_idx_data[b][j];
            }
            file << "\n";
        }
        
        file << "# Batch " << b << " info\n";
        file << args.info_data[b] << "\n";
    }
    
    file.close();
    std::cout << "Saved getf2 args to: " << filename << std::endl;
}

// Capture arguments before kernel call (for batched case with T* const* A)
template <typename T, typename I, typename INFO, typename U>
void captureArgs(const I m,
                 const I n,
                 U A,
                 const rocblas_stride shiftA,
                 const I lda,
                 const rocblas_stride strideA,
                 I* ipiv,  // ipiv is strided, not batched pointers
                 const rocblas_stride shiftP,
                 const rocblas_stride strideP,
                 INFO* info,
                 const I batch_count,
                 const bool pivot,
                 const I offset,
                 I* permut_idx,
                 const rocblas_stride stridePI,
                 dim3 grid,
                 dim3 block,
                 size_t lmemsize,
                 const std::string& dir) {
    
    Getf2CallArgs<T, I> args;
    args.m = m;
    args.n = n;
    args.lda = lda;
    args.batch_count = batch_count;
    args.offset = offset;
    args.shiftA = shiftA;
    args.strideA = strideA;
    args.shiftP = shiftP;
    args.strideP = strideP;
    args.stridePI = stridePI;
    args.pivot = pivot;
    args.grid_x = grid.x; args.grid_y = grid.y; args.grid_z = grid.z;
    args.block_x = block.x; args.block_y = block.y; args.block_z = block.z;
    args.lmemsize = lmemsize;
    
    // Allocate host memory
    args.A_data.resize(batch_count);
    args.ipiv_data.resize(batch_count);
    args.info_data.resize(batch_count);
    if (permut_idx) {
        args.permut_idx_data.resize(batch_count);
    }
    
    hipDeviceSynchronize();
    
    // Check if A is a batched pointer (T* const*) or strided pointer (T*)
    constexpr bool is_batched = std::is_pointer_v<std::remove_cv_t<std::remove_pointer_t<std::remove_cv_t<U>>>>;
    
    if constexpr (is_batched) {
        // Batched case: A is T* const*
        std::vector<T*> h_A_ptrs(batch_count);
        hipMemcpy(h_A_ptrs.data(), A, batch_count * sizeof(T*), hipMemcpyDeviceToHost);
        hipDeviceSynchronize();
        
        for (I b = 0; b < batch_count; ++b) {
            // Copy matrix A for this batch
            size_t A_size = m * lda;
            args.A_data[b].resize(A_size);
            hipMemcpy(args.A_data[b].data(), h_A_ptrs[b] + shiftA, 
                      A_size * sizeof(T), hipMemcpyDeviceToHost);
            
            // ipiv is strided (not batched pointers)
            args.ipiv_data[b].resize(n);
            hipMemcpy(args.ipiv_data[b].data(), ipiv + shiftP + b * strideP,
                      n * sizeof(I), hipMemcpyDeviceToHost);
            
            // Copy permut_idx if present
            if (permut_idx) {
                args.permut_idx_data[b].resize(m);
                hipMemcpy(args.permut_idx_data[b].data(), 
                          permut_idx + b * stridePI,
                          m * sizeof(I), hipMemcpyDeviceToHost);
            }
        }
    } else {
        // Strided case: A is T*
        for (I b = 0; b < batch_count; ++b) {
            size_t A_size = m * lda;
            args.A_data[b].resize(A_size);
            T* A_ptr = const_cast<T*>(reinterpret_cast<const T*>(A));
            hipMemcpy(args.A_data[b].data(), A_ptr + shiftA + b * strideA, 
                      A_size * sizeof(T), hipMemcpyDeviceToHost);
            
            args.ipiv_data[b].resize(n);
            hipMemcpy(args.ipiv_data[b].data(), ipiv + shiftP + b * strideP,
                      n * sizeof(I), hipMemcpyDeviceToHost);
            
            if (permut_idx) {
                args.permut_idx_data[b].resize(m);
                hipMemcpy(args.permut_idx_data[b].data(), 
                          permut_idx + b * stridePI,
                          m * sizeof(I), hipMemcpyDeviceToHost);
            }
        }
    }
    
    // Copy info
    std::vector<INFO> h_info(batch_count);
    hipMemcpy(h_info.data(), info, batch_count * sizeof(INFO), hipMemcpyDeviceToHost);
    for (I b = 0; b < batch_count; ++b) {
        args.info_data[b] = static_cast<I>(h_info[b]);
    }
    
    hipDeviceSynchronize();
    
    int& callCount = getCaptureCallCount();
    std::string filename = dir + "/getf2_args_call" + std::to_string(callCount) + ".txt";
    saveArgsToFile(args, filename);
    callCount++;
}

} // namespace getf2_capture

