#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <system_error>
#include <vector>
#include <pti_view.h>

#include "pti-adapter.h"
#include "utilities.h"

int main(int argc, char* argv[]) {
  auto return_value = EXIT_FAILURE;

  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " sycl_program [sycl_program_args...]"
              << '\n';
    return return_value;
  }

  try {
    auto args = std::vector<char*>(argv + 1, argv + argc);
    args.push_back(nullptr);

    utilities::SetEnv("LD_PRELOAD", utilities::GetPathToSharedObject(&PtiAdapterInit).c_str());

    PtiAdapterInit();

    return_value = utilities::LaunchProgram(args);

  } catch (const utilities::PosixErr& e) {
    std::cerr << "[FATAL] Message from system: " << e.what() << '\n';
  } catch (const std::system_error& e) {
    std::cerr << "[FATAL] Message from system: " << e.what() << '\n';
  } catch (...) {
    std::cerr << "[FATAL] Unknown Error" << '\n';
  }

  return return_value;
}
