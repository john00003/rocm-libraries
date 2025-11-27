// Copyright (c) 2025 Advanced Micro Devices, Inc.
// All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "../../example_utils.hpp"
#include <numeric>      // for std::partial_sum
#include <vector>

int main()
{
    // Prepare test data
    std::vector<int> host_input{6, 3, 5, 4, 1, 8, 2, 5, 4, 1};
    common::device_ptr<int> input(host_input);
    common::device_ptr<int> output(host_input.size());

    // Required temporary storage size
    size_t temp_storage_size = 0;
    HIP_CHECK(rocprim::inclusive_scan(nullptr,
                                      temp_storage_size,
                                      input.get(),
                                      output.get(),
                                      host_input.size()));

    // Allocate temporary storage
    common::device_ptr<void> temp_storage(temp_storage_size);

    // Perform the inclusive scan
    HIP_CHECK(rocprim::inclusive_scan(temp_storage.get(),
                                      temp_storage_size,
                                      input.get(),
                                      output.get(),
                                      host_input.size()));

    // Check for device-side errors
    HIP_CHECK(hipGetLastError());

    // Wait for the algorithm to finish
    HIP_CHECK(hipDeviceSynchronize());

    // Copy result to host
    auto result = output.load();

    // Compute expected result on the host
    std::vector<int> expected(host_input.size());
    std::partial_sum(host_input.begin(), host_input.end(), expected.begin());

    // Validate result
    for(size_t i = 0; i < expected.size(); ++i)
    {
        ASSERT_TRUE(result[i] == expected[i]);
    }

    return 0;
}
