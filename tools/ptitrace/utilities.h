#ifndef UTILITIES_H_
#define UTILITIES_H_
#include <dlfcn.h>
#include <spawn.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>

#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

extern "C" {

extern char** environ;
}

namespace utilities {

class PosixErr : public std::exception {
 public:
  explicit PosixErr(std::string_view err_msg) : what_msg_(err_msg) {}

  [[nodiscard]] inline const char* what() const noexcept override {
    return std::data(what_msg_);
  }

 private:
  std::string_view what_msg_;
};

template <typename T>
constexpr inline void CheckPosixRet(T ret_value) {
  static_assert(std::is_signed<T>::value);
  if (ret_value < 0) {
    throw PosixErr{strerror(errno)};
  }
}

constexpr inline auto kNsecInSec = 1'000'000'000;

inline uint32_t GetPid() {
  auto ret = getpid();
  CheckPosixRet(ret);
  return ret;
}

inline uint32_t GetTid() {
#if defined(SYS_gettid)
  auto ret = syscall(SYS_gettid);
  CheckPosixRet(ret);
  return ret;
#else
#error "SYS_gettid is unavailable on this system"
#endif
}

inline uint64_t GetTime(clockid_t id) {
  timespec ts{0, 0};
  auto status = clock_gettime(id, &ts);
  CheckPosixRet(status);
  return ts.tv_sec * kNsecInSec + ts.tv_nsec;
}

inline void SetEnv(const char* name, const char* value) {
  auto status = setenv(name, value, 1);
  CheckPosixRet(status);
}

inline std::string GetEnv(const char* name) {
  auto* result = secure_getenv(name);
  if(!result) {
    return std::string{};
  }

  return std::string{result};
}

inline void PrintEnv(const char* name) {
 std::cout << '\t' << GetEnv(name) << '\n';
}


template <typename T>
inline std::string GetPathToSharedObject(T address) {
  static_assert(std::is_pointer<T>::value);
  Dl_info info{nullptr, nullptr, nullptr, nullptr};
  auto status = dladdr(reinterpret_cast<const void*>(address), &info);
  CheckPosixRet(status);
  return std::string{info.dli_fname};
}

template <typename T>
inline auto Spawn(const T& prog_and_args) {
  pid_t cpid = 0;
  posix_spawn_file_actions_t* file_actions_p = nullptr;
  posix_spawnattr_t* attrp = nullptr;
  const auto result = posix_spawn(&cpid, prog_and_args.front(), file_actions_p,
                                  attrp, std::data(prog_and_args), environ);
  if (result) {
    throw std::system_error(result, std::generic_category(),
                            "Failed to spawn process");
  }
  return cpid;
}

template <typename PidType>
inline auto Wait(PidType pid_val, int options) {
  int status = 0;
  auto result = waitpid(pid_val, &status, options);
  if (result < 0) {
    throw std::system_error(result, std::generic_category(),
                            "Failed to wait for process");
  }

  return status;
}

template <typename PidType>
inline auto CheckExitStatus(PidType pid_val, int options) {
  int return_value = 1;
  std::optional<int> status = std::nullopt;
  do {
    status = Wait(pid_val, options);
    if (WIFEXITED(*status)) {
      return_value = WEXITSTATUS(*status);
    } else if (WIFSIGNALED(*status)) {
      return_value = WTERMSIG(*status);
      std::cout << "Processed killed by signal: " << return_value << '\n';
    } else if (WIFSTOPPED(*status)) {
      std::cout << "Processed stopped by signal: " << WSTOPSIG(*status) << '\n';
    } else if (WIFCONTINUED(*status)) {
      std::cout << "Continued..." << '\n';
    }
  } while (!WIFEXITED(*status) && !WIFSIGNALED(*status));

  return return_value;
}

template <typename T>
inline auto LaunchProgram(const T& prog_and_args) {
  auto process_id = Spawn(prog_and_args);
  return CheckExitStatus(process_id, WUNTRACED | WCONTINUED);
}

template<typename T>
inline void* WriteToShmem(T easy_type) {
  int prot = PROT_READ | PROT_WRITE;
  int flags = MAP_SHARED | MAP_ANONYMOUS;
  auto* result = mmap(nullptr, sizeof(T), prot, flags, -1, 0);
  if(result == MAP_FAILED) {
    throw std::runtime_error("Failed to alloc shmem");
  }
  return std::memcpy(result, &easy_type, sizeof(T));
}

template<typename T>
inline T ReadFromShmem(void* shmem_ptr) {
  T data{};
  std::memcpy(&data, shmem_ptr, sizeof(T));
  return data;
}

inline auto Fork() {
  auto ret = fork();
  CheckPosixRet(ret);
  return ret;
}

inline void Exec(const std::vector<char*>& prog_args) {
  auto ret = execvp(prog_args.front(), std::data(prog_args));
  CheckPosixRet(ret);
}

inline auto Wait(int* return_val) {
  auto ret = wait(return_val);
  CheckPosixRet(ret);
  return ret;
}

}  // namespace utilities
#endif
