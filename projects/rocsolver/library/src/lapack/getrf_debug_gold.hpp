/************************************************************************
 * Debug utilities for GETRF gold comparison
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 * *************************************************************************/

#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <hip/hip_runtime.h>

namespace rocsolver_debug {

// Structure to hold gold matrix data for one batch
template <typename T>
struct GoldMatrix {
    std::vector<T> data;           // Matrix data (row-major, m x n)
    int m = 0;
    int n = 0;
};

// Parse a comma-separated line of values
template <typename T>
std::vector<T> parseCSVLine(const std::string& line) {
    std::vector<T> values;
    std::stringstream ss(line);
    std::string token;
    
    while (std::getline(ss, token, ',')) {
        // Trim whitespace
        size_t start = token.find_first_not_of(" \t");
        size_t end = token.find_last_not_of(" \t");
        if (start != std::string::npos && end != std::string::npos) {
            token = token.substr(start, end - start + 1);
        }
        
        if (!token.empty()) {
            if constexpr (std::is_integral_v<T>) {
                values.push_back(static_cast<T>(std::stoi(token)));
            } else {
                values.push_back(static_cast<T>(std::stof(token)));
            }
        }
    }
    return values;
}

// Read gold matrix from file
// Format: Each line is a row of matrix data (comma-separated values)
template <typename T>
bool readGoldMatrix(const std::string& filepath, GoldMatrix<T>& gold) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open gold file: " << filepath << std::endl;
        return false;
    }
    
    std::string line;
    gold.m = 0;
    gold.n = 0;
    gold.data.clear();
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        // All lines are matrix data rows
        auto rowValues = parseCSVLine<T>(line);
        if (!rowValues.empty()) {
            if (gold.n == 0) {
                gold.n = rowValues.size();  // Set column count from first row
            }
            gold.data.insert(gold.data.end(), rowValues.begin(), rowValues.end());
            gold.m++;
        }
    }
    
    file.close();
    
    std::cout << "Read gold matrix from " << filepath << ": " << gold.m << "x" << gold.n 
              << " (" << gold.data.size() << " elements)" << std::endl;
    
    return gold.m > 0 && gold.n > 0;
}

// Read gold matrices for all batches
template <typename T>
bool readAllGoldMatrices(const std::string& baseDir, 
                         const std::string& prefix,
                         int batchCount,
                         std::vector<GoldMatrix<T>>& goldMatrices) {
    goldMatrices.clear();
    goldMatrices.resize(batchCount);
    
    for (int b = 0; b < batchCount; b++) {
        std::string filepath = baseDir + "/" + prefix + "__" + std::to_string(b);
        if (!readGoldMatrix(filepath, goldMatrices[b])) {
            return false;
        }
    }
    return true;
}

// Compare a single host matrix with gold matrix
template <typename T>
bool compareSingleBatch(const T* h_A,         // Host pointer to matrix
                        int lda,              // Leading dimension of A
                        int m,                // Number of rows  
                        int n,                // Number of columns
                        int batchIdx,
                        const GoldMatrix<T>& gold,
                        const std::string& checkpointName,
                        int& mismatchCount,
                        int maxMismatches,
                        T tolerance) {
    bool batchMatch = true;
    
    for (int i = 0; i < m && mismatchCount < maxMismatches; i++) {
        for (int j = 0; j < n && mismatchCount < maxMismatches; j++) {
            // Device matrix is column-major: A[i + j*lda]
            // Gold matrix is stored row-major in the file: gold.data[i * n + j]
            T deviceVal = h_A[i + j * lda];
            T goldVal = gold.data[i * gold.n + j];
            
            T diff = std::abs(deviceVal - goldVal);
            T maxVal = std::max(std::abs(deviceVal), std::abs(goldVal));
            // T relErr = (maxVal > static_cast<T>(1e-10)) ? diff / maxVal : diff;
            T relErr = diff;
            
            if (relErr > tolerance && diff > tolerance) {
                if (mismatchCount == 0) {
                    std::cerr << "\n=== MISMATCH at checkpoint: " << checkpointName << " ===" << std::endl;
                }
                std::cerr << "  Batch " << batchIdx << ", A[" << i << "," << j << "]: "
                          << "device=" << deviceVal << ", gold=" << goldVal 
                          << ", diff=" << diff << ", relErr=" << relErr << std::endl;
                batchMatch = false;
                mismatchCount++;
            }
        }
    }
    return batchMatch;
}

// Compare device matrix with gold matrix (BATCHED version - array of pointers)
template <typename T>
bool compareWithGoldBatched(T* const* d_Aarray,  // Device array of pointers
                            rocblas_stride shiftA,  // Shift into each matrix
                            int lda,                // Leading dimension of A
                            int m,                  // Number of rows
                            int n,                  // Number of columns
                            int batchCount,
                            const std::vector<GoldMatrix<T>>& goldMatrices,
                            const std::string& checkpointName,
                            T tolerance = static_cast<T>(1e-4)) {
    
    // First, copy the array of device pointers to host
    std::vector<T*> h_Aarray(batchCount);
    hipError_t err = hipMemcpy(h_Aarray.data(), d_Aarray, batchCount * sizeof(T*), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error: hipMemcpy failed copying pointer array" << std::endl;
        return false;
    }
    
    hipDeviceSynchronize();
    
    bool allMatch = true;
    int maxMismatches = 10;
    int mismatchCount = 0;
    
    // Allocate host buffer for one matrix
    size_t matrixSize = static_cast<size_t>(lda) * n;
    std::vector<T> h_A(matrixSize);
    
    for (int b = 0; b < batchCount && mismatchCount < maxMismatches; b++) {
        // Copy this batch's matrix from device to host
        T* d_batchA = h_Aarray[b] + shiftA;
        err = hipMemcpy(h_A.data(), d_batchA, matrixSize * sizeof(T), hipMemcpyDeviceToHost);
        if (err != hipSuccess) {
            std::cerr << "Error: hipMemcpy failed for batch " << b << std::endl;
            return false;
        }
        
        hipDeviceSynchronize();
        
        const GoldMatrix<T>& gold = goldMatrices[b];
        if (!compareSingleBatch(h_A.data(), lda, m, n, b, gold, 
                                checkpointName, mismatchCount, maxMismatches, tolerance)) {
            allMatch = false;
        }
    }
    
    if (allMatch) {
        std::cout << "=== MATCH at checkpoint: " << checkpointName << " ===" << std::endl;
    } else if (mismatchCount >= maxMismatches) {
        std::cerr << "  ... (more mismatches suppressed)" << std::endl;
    }
    
    return allMatch;
}

// Compare device matrix with gold matrix (STRIDED version - single pointer with stride)
template <typename T>
bool compareWithGoldStrided(const T* d_A,           // Device pointer to matrix A
                            int lda,                // Leading dimension of A
                            int m,                  // Number of rows
                            int n,                  // Number of columns
                            rocblas_stride strideA, // Stride between batches
                            int batchCount,
                            const std::vector<GoldMatrix<T>>& goldMatrices,
                            const std::string& checkpointName,
                            T tolerance = static_cast<T>(1e-4)) {
    
    // Allocate host memory for all batches
    size_t totalSize = strideA * batchCount;
    std::vector<T> h_A(totalSize);
    
    // Copy from device to host
    hipError_t err = hipMemcpy(h_A.data(), d_A, totalSize * sizeof(T), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error: hipMemcpy failed in compareWithGoldStrided" << std::endl;
        return false;
    }
    
    hipDeviceSynchronize();
    
    bool allMatch = true;
    int maxMismatches = 10;
    int mismatchCount = 0;
    
    for (int b = 0; b < batchCount && mismatchCount < maxMismatches; b++) {
        const T* batchA = h_A.data() + b * strideA;
        const GoldMatrix<T>& gold = goldMatrices[b];
        
        if (!compareSingleBatch(batchA, lda, m, n, b, gold,
                                checkpointName, mismatchCount, maxMismatches, tolerance)) {
            allMatch = false;
        }
    }
    
    if (allMatch) {
        std::cout << "=== MATCH at checkpoint: " << checkpointName << " ===" << std::endl;
    } else if (mismatchCount >= maxMismatches) {
        std::cerr << "  ... (more mismatches suppressed)" << std::endl;
    }
    
    return allMatch;
}

} // namespace rocsolver_debug

