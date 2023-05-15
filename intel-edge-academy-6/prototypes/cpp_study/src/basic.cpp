#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#define MAX_I 100000
#define TOTAL_TEST_COUNT 10000

void benchmark_pointer_operations() {
  // int nArr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};  // Memory Debugging . 4 Byte
  // (int) * 10. Little Endian. // refer to Autos or Watch 1 Tab for watching
  // value
  int nArr[MAX_I] = {};
  for (size_t i = 0; i < std::size(nArr); i++) {
    nArr[i] = i + 1;
  }

  int *pArr = nArr;
  int length = std::size(nArr);
  int sum = 0;
  std::chrono::steady_clock::time_point begin;
  std::chrono::steady_clock::time_point end;
  long elapsed_time_sum[3] = {
      0,
  };

  for (size_t test_count = 0; test_count < TOTAL_TEST_COUNT; test_count++) {
    pArr = nArr;
    sum = 0;
    begin = std::chrono::steady_clock::now();
    for (size_t i = 0; i < length; i++) {
      sum += *(pArr + i);
    }
    end = std::chrono::steady_clock::now();
    elapsed_time_sum[0] +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
            .count();

    pArr = nArr;
    sum = 0;
    begin = std::chrono::steady_clock::now();
    for (size_t i = 0; i < length; i++) {
      sum += pArr[i];
    }
    end = std::chrono::steady_clock::now();
    elapsed_time_sum[1] +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
            .count();

    pArr = nArr;
    sum = 0;
    begin = std::chrono::steady_clock::now();
    for (size_t i = 0; i < length; i++) {
      sum += *(pArr++);
    }
    end = std::chrono::steady_clock::now();
    elapsed_time_sum[2] +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
            .count();
  }

  std::cout << "Time difference = " << elapsed_time_sum[0] / TOTAL_TEST_COUNT
            << "[ns]" << std::endl;
  std::cout << "Time difference = " << elapsed_time_sum[1] / TOTAL_TEST_COUNT
            << "[ns]" << std::endl;
  std::cout << "Time difference = " << elapsed_time_sum[2] / TOTAL_TEST_COUNT
            << "[ns]" << std::endl;
}