#pragma once
#include <cstddef>
#include <cstdlib>
#include <dlfcn.h>

#include <string>

namespace lingodb::runtime {

inline void* getCudaRtHandle() {
   static void* handle = nullptr;
   static bool initialized = false;
   if (!initialized) {
      initialized = true;
      if (const char* cudaPath = std::getenv("CUDA_PATH")) {
         std::string p = std::string(cudaPath) + "/lib64/libcudart.so";
         handle = dlopen(p.c_str(), RTLD_LAZY | RTLD_GLOBAL);
      }
      if (!handle) {
         if (const char* cudaHome = std::getenv("CUDA_HOME")) {
            std::string p = std::string(cudaHome) + "/lib64/libcudart.so";
            handle = dlopen(p.c_str(), RTLD_LAZY | RTLD_GLOBAL);
         }
      }
      if (!handle) handle = dlopen("libcudart.so", RTLD_LAZY | RTLD_GLOBAL);
      if (!handle) handle = dlopen("libcudart.so.12", RTLD_LAZY | RTLD_GLOBAL);
      if (!handle) handle = dlopen("libcudart.so.11", RTLD_LAZY | RTLD_GLOBAL);
      if (!handle) handle = dlopen("/usr/local/cuda/lib64/libcudart.so", RTLD_LAZY | RTLD_GLOBAL);
      if (!handle) handle = dlopen("/usr/local/cuda/lib/libcudart.so", RTLD_LAZY | RTLD_GLOBAL);
   }
   return handle;
}

inline void* cudaAllocManaged(size_t size) {
   using MallocManagedFn = int (*)(void**, size_t, unsigned int);
   static MallocManagedFn fn = nullptr;
   static bool initialized = false;
   if (!initialized) {
      initialized = true;
      void* handle = getCudaRtHandle();
      if (handle) {
         fn = reinterpret_cast<MallocManagedFn>(dlsym(handle, "cudaMallocManaged"));
      }
   }
   if (fn) {
      void* ptr = nullptr;
      if (fn(&ptr, size, 1 /* cudaMemAttachGlobal */) == 0 && ptr) {
         return ptr;
      }
   }
   return std::malloc(size);
}

inline void cudaFreeManaged(void* ptr) {
   if (!ptr) return;
   using FreeFn = int (*)(void*);
   static FreeFn fn = nullptr;
   static bool initialized = false;
   if (!initialized) {
      initialized = true;
      void* handle = getCudaRtHandle();
      if (handle) {
         fn = reinterpret_cast<FreeFn>(dlsym(handle, "cudaFree"));
      }
   }
   if (fn && fn(ptr) == 0) {
      return;
   }
   std::free(ptr);
}

} // namespace lingodb::runtime

