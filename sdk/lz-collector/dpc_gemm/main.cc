//==============================================================
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <string.h>

#include "lz_collector.h"
#include <sycl/sycl.hpp>
#include <cstdlib>
#include <memory>
#include <exception>

#define A_VALUE 0.128f
#define B_VALUE 0.256f
#define MAX_EPS 1.0e-4f
#define NSEC_IN_SEC 1'000'000'00

#define PTI_ASSERT(X) do { \
  if (!(X)) \
    throw std::runtime_error("PTI_ASSERT!!"); \
} \
  while(0)

void StartTracing() {
}

void StopTracing() {
}

static float Check(const std::vector<float> &a, float value) {
  PTI_ASSERT(value > MAX_EPS);

  float eps = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    eps += fabs((a[i] - value) / value);
  }

  return eps / a.size();
}

void GEMM(const float *a, const float *b, float *c, unsigned size,
          sycl::id<2> id) {
  int i = id.get(0);
  int j = id.get(1);
  float sum = 0.0f;
  for (unsigned k = 0; k < size; ++k) {
    sum += a[i * size + k] * b[k * size + j];
  }
  c[i * size + j] = sum;
}

static float RunAndCheck(sycl::queue queue, const std::vector<float> &a,
                         const std::vector<float> &b, std::vector<float> &c,
                         unsigned size, float expected_result) {
  PTI_ASSERT(size > 0);
  PTI_ASSERT(a.size() == size * size);
  PTI_ASSERT(b.size() == size * size);
  PTI_ASSERT(c.size() == size * size);

  double time = 0.0;

  try {
    sycl::buffer<float, 1> a_buf(a.data(), a.size());
    sycl::buffer<float, 1> b_buf(b.data(), b.size());
    sycl::buffer<float, 1> c_buf(c.data(), c.size());

    sycl::event event = queue.submit([&](sycl::handler &cgh) {
      auto a_acc = a_buf.get_access<sycl::access::mode::read>(cgh);
      auto b_acc = b_buf.get_access<sycl::access::mode::read>(cgh);
      auto c_acc = c_buf.get_access<sycl::access::mode::write>(cgh);

      cgh.parallel_for<class __GEMM>(
          sycl::range<2>(size, size), [=](sycl::id<2> id) {
            auto a_acc_ptr = a_acc.get_multi_ptr<sycl::access::decorated::no>();
            auto b_acc_ptr = b_acc.get_multi_ptr<sycl::access::decorated::no>();
            auto c_acc_ptr = c_acc.get_multi_ptr<sycl::access::decorated::no>();
            GEMM(a_acc_ptr.get(), b_acc_ptr.get(), c_acc_ptr.get(), size, id);
          });
    });
    queue.wait_and_throw();

    auto start =
        event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end =
        event.get_profiling_info<sycl::info::event_profiling::command_end>();
    time = static_cast<double>(end - start) / NSEC_IN_SEC;
  } catch (const sycl::exception& e) {
    std::cout << "[ERROR] " << e.what() << std::endl;
    throw;
  }

  std::cout << "Matrix multiplication time: " << time << " sec" << std::endl;

  return Check(c, expected_result);
}

static void Compute(sycl::queue queue, const std::vector<float> &a,
                    const std::vector<float> &b, std::vector<float> &c,
                    unsigned size, unsigned repeat_count,
                    float expected_result) {
  for (unsigned i = 0; i < repeat_count; ++i) {
    float eps = RunAndCheck(queue, a, b, c, size, expected_result);
    std::cout << "Results are " << ((eps < MAX_EPS) ? "" : "IN")
              << "CORRECT with accuracy: " << eps << std::endl;
  }
}

const unsigned max_size = 8192;
const unsigned min_size = 32;

void Usage(const char* name) {

  std::cout << " Calculating floating point matrix multiply on gpu\n";
  std::cout << name << " [ [gpu|cpu|host, default=gpu],  [matrix size, default=1024, max="
            << max_size << "], [repetition count, default=4]] \n";
}

int main(int argc, char *argv[]) {
  int exit_code = EXIT_SUCCESS;
  // start tracing early enables to capture nodes creation at piProgramCreate
  //  and Kernel Task sycl file/line info is captured, as exampple shows at a Node Creation
  StartTracing();
  unsigned repeat_count = 4;
  unsigned size = 1024;
  sycl::device dev;
  try {
    dev = sycl::device(sycl::gpu_selector_v);
    if (argc > 1 && strcmp(argv[1], "cpu") == 0) {
      dev = sycl::device(sycl::cpu_selector_v);
      std::cerr << "PTI doesn't support cpu profiling yet" << '\n';
      std::exit(EXIT_FAILURE);
    } else if (argc > 1 && strcmp(argv[1], "host") == 0) {
      dev = sycl::device(sycl::default_selector_v);
      std::cerr << "PTI doesn't support host profiling yet" << '\n';
      std::exit(EXIT_FAILURE);
    }

    unsigned temp = size;
    if (argc > 2) {
      temp = std::stoul(argv[2]);
      size = (temp < min_size) ? min_size :
                    (temp > max_size) ?  max_size : temp;
    }

    if (argc > 3) {
      temp = std::stoul(argv[3]);
      repeat_count = (temp < 1) ? 1 : temp;
    }
  } catch (const sycl::exception &e) {
    Usage(argv[0]);
    std::cerr << "Error: Exception caught while executing SYCL " << e.what() << '\n';
    std::cerr << "Unable to select valid sycl device" << '\n';
    return EXIT_FAILURE;
  } catch(...) {
    Usage(argv[0]);
    return EXIT_FAILURE;
  }


  sycl::property_list prop_list{sycl::property::queue::enable_profiling(), sycl::ext::intel::property::queue::no_immediate_command_list()};
  sycl::queue queue(dev, sycl::async_handler{}, prop_list);

  std::cout << "DPC++ Matrix Multiplication (matrix size: " << size << " x "
            << size << ", repeats " << repeat_count << " times)" << std::endl;
  std::cout << "Target device: "
            << queue.get_info<sycl::info::queue::device>()
                  .get_info<sycl::info::device::name>()
            << std::endl;

  std::vector<float> a(size * size, A_VALUE);
  std::vector<float> b(size * size, B_VALUE);
  std::vector<float> c(size * size, 0.0f);

  try {
    LzCollector collector{};
    auto start = std::chrono::steady_clock::now();
    float expected_result = A_VALUE * B_VALUE * size;
    Compute(queue, a, b, c, size, repeat_count, expected_result);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<float> time = end - start;
    std::cout << "Total execution time with tracing: " << time.count() << " sec"
              << std::endl;

    start = std::chrono::steady_clock::now();
    expected_result = A_VALUE * B_VALUE * size;

    Compute(std::move(queue), a, b, c, size, repeat_count, expected_result);
    end = std::chrono::steady_clock::now();
    time = end - start;
    collector.PrintResults();

    std::cout << "Total execution time without tracing: " << time.count()
              << " sec" << std::endl;
  } catch (const sycl::exception &e) {
    std::cerr << "Error: Exception while executing SYCL " << e.what() << '\n';
    std::cerr << "\tError code: " << e.code().value()
              << "\n\tCategory: " << e.category().name()
              << "\n\tMessage: " << e.code().message() << '\n';
    exit_code = EXIT_FAILURE;
  } catch (const std::exception &e) {
    std::cerr << "Error: Exception caught " << e.what() << '\n';
    exit_code = EXIT_FAILURE;
  } catch (...) {
    std::cerr << "Error: Unknown exception caught." << '\n';
    exit_code = EXIT_FAILURE;
  }

  return exit_code;
}
