#ifndef LINGODB_RUNTIME_GPUHASHTABLE_HPP
#define LINGODB_RUNTIME_GPUHASHTABLE_HPP

#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <cstring>
#include "lingodb/runtime/CudaMemory.h"

#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif

namespace lingodb::runtime {

constexpr int64_t GPU_HT_EMPTY_KEY = -1LL;

template <typename T>
inline __device__ __host__ T gpuAtomicCAS(T* address, T compare, T val) {
#if defined(__CUDA_ARCH__) || defined(__ACPP_DEVICE_CODE__)
   return static_cast<T>(atomicCAS(reinterpret_cast<unsigned long long int*>(address),
                                    static_cast<unsigned long long int>(compare),
                                    static_cast<unsigned long long int>(val)));
#elif defined(__clang__) || defined(__GNUC__)
   T expected = compare;
   __atomic_compare_exchange_n(address, &expected, val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
   return expected;
#else
   return static_cast<T>(atomicCAS(reinterpret_cast<unsigned long long int*>(address),
                                    static_cast<unsigned long long int>(compare),
                                    static_cast<unsigned long long int>(val)));
#endif
}

struct GPUHashTable {
   int64_t* keys{nullptr};
   int64_t* values{nullptr};
   uint64_t capacity{0};
   uint64_t mask{0};
   uint64_t num_entries{0};

   static uint64_t nextPow2(uint64_t v) {
      if (v == 0) return 1;
      v--;
      v |= v >> 1;
      v |= v >> 2;
      v |= v >> 4;
      v |= v >> 8;
      v |= v >> 16;
      v |= v >> 32;
      v++;
      return v;
   }

   static GPUHashTable* create(uint64_t estimated_elements) {
      auto* ht = new GPUHashTable();
      ht->capacity = std::max(nextPow2(estimated_elements * 2), static_cast<uint64_t>(1024));
      ht->mask = ht->capacity - 1;
      ht->num_entries = 0;

      ht->keys = reinterpret_cast<int64_t*>(cudaAllocManaged(ht->capacity * sizeof(int64_t)));
      ht->values = reinterpret_cast<int64_t*>(cudaAllocManaged(ht->capacity * sizeof(int64_t)));

      memset(ht->keys, 0xFF, ht->capacity * sizeof(int64_t));
      memset(ht->values, 0, ht->capacity * sizeof(int64_t));

      return ht;
   }

   static void destroy(GPUHashTable* ht) {
      if (!ht) return;
      if (ht->keys) cudaFreeManaged(ht->keys);
      if (ht->values) cudaFreeManaged(ht->values);
      delete ht;
   }

   static inline __device__ __host__ uint64_t hash(int64_t key, uint64_t mask) {
      return (static_cast<uint64_t>(key) * 0x9E3779B97F4A7C15ull) & mask;
   }

   inline __device__ __host__ void insert(int64_t key, int64_t tuple_id) {
      uint64_t slot = hash(key, mask);
      for (;; slot = (slot + 1) & mask) {
         unsigned long long* slot_ptr = reinterpret_cast<unsigned long long*>(&keys[slot]);
         unsigned long long expected = static_cast<unsigned long long>(GPU_HT_EMPTY_KEY);
         unsigned long long desired = static_cast<unsigned long long>(key);
         unsigned long long old = gpuAtomicCAS(slot_ptr, expected, desired);
         if (old == expected) {
            values[slot] = tuple_id;
            break;
         } else if (old == desired) {
            break;
         }
      }
   }

   inline __device__ __host__ int64_t lookup(int64_t key) const {
      uint64_t slot = hash(key, mask);
      for (;; slot = (slot + 1) & mask) {
         int64_t k = keys[slot];
         if (k == key) {
            return values[slot];
         }
         if (k == GPU_HT_EMPTY_KEY) {
            return -1LL;
         }
      }
   }

   void buildParallel(const int64_t* keys_in, int64_t num_keys, int64_t base_tuple_id = 0) {
      if (!keys_in || num_keys <= 0) return;
      for (int64_t i = 0; i < num_keys; ++i) {
         insert(keys_in[i], base_tuple_id + i);
      }
   }

   void probeBatch(const int64_t* keys_in, int64_t* matches_out, size_t n) const {
      if (!keys_in || !matches_out || n == 0) return;
      for (size_t i = 0; i < n; ++i) {
         matches_out[i] = lookup(keys_in[i]);
      }
   }
};

} // namespace lingodb::runtime

#endif // LINGODB_RUNTIME_GPUHASHTABLE_HPP
