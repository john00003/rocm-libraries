/*******************************************************************************
 * Gold matrix device loader for in-kernel verification of getf2_small_kernel
 * 
 * This header provides functions to:
 * 1. Load all 9 gold matrices (3 checkpoints x 3 batches) from files to device
 * 2. Support in-kernel comparison against expected results
 ******************************************************************************/

#pragma once

#include <hip/hip_runtime.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>

namespace getf2_gold {

// Constants for gold matrices
constexpr int GOLD_MATRIX_ROWS = 70;
constexpr int GOLD_MATRIX_COLS = 70;
constexpr int GOLD_NUM_CHECKPOINTS = 3;  // k0, k24, k48
constexpr int GOLD_NUM_BATCHES = 3;      // batch 0, 1, 2
constexpr int GOLD_TOTAL_MATRICES = GOLD_NUM_CHECKPOINTS * GOLD_NUM_BATCHES;
constexpr size_t GOLD_MATRIX_SIZE = GOLD_MATRIX_ROWS * GOLD_MATRIX_COLS;
constexpr size_t GOLD_TOTAL_SIZE = GOLD_TOTAL_MATRICES * GOLD_MATRIX_SIZE;

// Checkpoint names
static const char* CHECKPOINT_NAMES[] = {"k0", "k24", "k48"};

/**
 * Parse a CSV line into a vector of floats
 */
template <typename T>
std::vector<T> parseGoldLine(const std::string& line) {
    std::vector<T> values;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ',')) {
        size_t start = token.find_first_not_of(" \t");
        size_t end = token.find_last_not_of(" \t");
        if (start != std::string::npos && end != std::string::npos) {
            token = token.substr(start, end - start + 1);
            try {
                values.push_back(static_cast<T>(std::stod(token)));
            } catch (...) {
                // Skip invalid tokens
            }
        }
    }
    return values;
}

/**
 * Read a single gold matrix from file (row-major format)
 * Returns true on success
 */
template <typename T>
bool readGoldMatrixFile(const std::string& filepath, 
                        T* matrix_out,
                        int expected_rows = GOLD_MATRIX_ROWS,
                        int expected_cols = GOLD_MATRIX_COLS) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open gold file: " << filepath << std::endl;
        return false;
    }
    
    std::string line;
    int row = 0;
    while (std::getline(file, line) && row < expected_rows) {
        if (line.empty()) continue;
        
        std::vector<T> values = parseGoldLine<T>(line);
        if (values.size() < static_cast<size_t>(expected_cols)) {
            std::cerr << "Warning: Gold file " << filepath << " row " << row 
                      << " has only " << values.size() << " values, expected " 
                      << expected_cols << std::endl;
        }
        
        // Copy to output (row-major)
        for (int col = 0; col < expected_cols && col < static_cast<int>(values.size()); ++col) {
            matrix_out[row * expected_cols + col] = values[col];
        }
        row++;
    }
    
    if (row < expected_rows) {
        std::cerr << "Warning: Gold file " << filepath << " has only " << row 
                  << " rows, expected " << expected_rows << std::endl;
    }
    
    return row > 0;
}

/**
 * Load all 9 gold matrices to device memory
 * 
 * Memory layout:
 *   gold_ptr[checkpoint * 3 + batch][row * cols + col]
 * where checkpoint: 0=k0, 1=k24, 2=k48
 *       batch: 0, 1, 2
 *       row, col: matrix indices (row-major)
 * 
 * @param gold_dir Directory containing gold-GETF2-* files
 * @param d_gold_ptr Output: device pointer to all gold matrices
 * @return true on success
 */
template <typename T>
bool loadAllGoldToDevice(const char* gold_dir, T** d_gold_ptr) {
    // Allocate host memory for all matrices
    std::vector<T> h_all_gold(GOLD_TOTAL_SIZE, T(0));
    
    // Read each gold file
    for (int checkpoint = 0; checkpoint < GOLD_NUM_CHECKPOINTS; ++checkpoint) {
        for (int batch = 0; batch < GOLD_NUM_BATCHES; ++batch) {
            // Build filename: gold-GETF2-kX__B
            std::string filename = std::string(gold_dir) + "/gold-GETF2-" 
                                 + CHECKPOINT_NAMES[checkpoint] + "__" 
                                 + std::to_string(batch);
            
            // Calculate offset in the big array
            size_t matrix_idx = checkpoint * GOLD_NUM_BATCHES + batch;
            T* matrix_ptr = h_all_gold.data() + matrix_idx * GOLD_MATRIX_SIZE;
            
            if (!readGoldMatrixFile<T>(filename, matrix_ptr)) {
                std::cerr << "Failed to read gold matrix: " << filename << std::endl;
                return false;
            }
            
            std::cout << "Loaded gold matrix: " << filename << std::endl;
        }
    }
    
    // Allocate device memory
    hipError_t err = hipMalloc(d_gold_ptr, GOLD_TOTAL_SIZE * sizeof(T));
    if (err != hipSuccess) {
        std::cerr << "hipMalloc failed for gold matrices: " << hipGetErrorString(err) << std::endl;
        return false;
    }
    
    // Copy to device
    err = hipMemcpy(*d_gold_ptr, h_all_gold.data(), GOLD_TOTAL_SIZE * sizeof(T), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "hipMemcpy failed for gold matrices: " << hipGetErrorString(err) << std::endl;
        hipFree(*d_gold_ptr);
        *d_gold_ptr = nullptr;
        return false;
    }
    
    std::cout << "Successfully loaded " << GOLD_TOTAL_MATRICES << " gold matrices to device ("
              << GOLD_TOTAL_SIZE * sizeof(T) << " bytes)" << std::endl;
    
    return true;
}

/**
 * Free device gold matrices
 */
template <typename T>
void freeGoldDevice(T* d_gold_ptr) {
    if (d_gold_ptr) {
        hipFree(d_gold_ptr);
    }
}

/**
 * Device function to get pointer to specific gold matrix
 * 
 * @param d_gold_ptr Base pointer to all gold matrices on device
 * @param checkpoint Which checkpoint (0=k0, 1=k24, 2=k48)
 * @param batch Batch index (0, 1, 2)
 * @return Pointer to start of the specific gold matrix (row-major)
 */
template <typename T>
__device__ __forceinline__ 
T* getGoldMatrix(T* d_gold_ptr, int checkpoint, int batch) {
    size_t matrix_idx = checkpoint * GOLD_NUM_BATCHES + batch;
    return d_gold_ptr + matrix_idx * GOLD_MATRIX_SIZE;
}

/**
 * Device function to get element from gold matrix
 * Gold matrices are stored in row-major format
 * 
 * @param gold_matrix Pointer to gold matrix
 * @param row Row index in full 70x70 matrix
 * @param col Column index in full 70x70 matrix
 * @return Element value
 */
template <typename T>
__device__ __forceinline__
T getGoldElement(T* gold_matrix, int row, int col) {
    return gold_matrix[row * GOLD_MATRIX_COLS + col];
}

/**
 * Device function to compare computed value against gold
 * Prints mismatch details and optionally aborts
 * 
 * @param computed Value computed by kernel
 * @param gold_matrix Pointer to gold matrix (row-major)
 * @param gold_row Row in gold matrix (with offset applied)
 * @param gold_col Column in gold matrix (with offset applied)
 * @param tolerance Comparison tolerance
 * @param batch_id Batch index for error reporting
 * @param checkpoint Checkpoint index for error reporting
 * @param myrow Thread's row for error reporting
 * @param j Column index in rA for error reporting
 * @return true if match, false if mismatch
 */
template <typename T>
__device__ __forceinline__
bool compareGoldValue(T computed, T* gold_matrix, int gold_row, int gold_col,
                      T tolerance, int batch_id, int checkpoint, int myrow, int j) {
    T expected = getGoldElement(gold_matrix, gold_row, gold_col);
    T diff = computed - expected;
    if (diff < 0) diff = -diff;  // abs
    
    T scale = (expected < 0 ? -expected : expected);
    if (scale < T(1)) scale = T(1);
    
    if (diff > tolerance * scale) {
        printf("MISMATCH: checkpoint=%d batch=%d myrow=%d j=%d: "
               "computed=%f expected=%f (gold[%d,%d]) diff=%f\n",
               checkpoint, batch_id, myrow, j,
               (float)computed, (float)expected, gold_row, gold_col, (float)diff);
        return false;
    }
    return true;
}

} // namespace getf2_gold

