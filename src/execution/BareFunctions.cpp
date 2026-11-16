#include "lingodb/execution/Backend.h"
#include "lingodb/runtime/helpers.h"

#include <iostream>
extern "C" lingodb::runtime::VarLen32 createVarLen32(uint8_t* ptr, uint32_t len); //NOLINT(clang-diagnostic-return-type-c-linkage)
extern "C" uint64_t hashVarLenData(lingodb::runtime::VarLen32 str);
extern "C" void dumpString(lingodb::runtime::VarLen32 str) {
   std::cout << "string(\"" << str.str() << "\")" << std::endl;
}
extern "C" void dumpI64(int64_t i) {
   std::cout << "int(" << i << ")" << std::endl;
}
extern "C" void dumpF64(double d) {
   std::cout << "float(" << d << ")" << std::endl;
}
extern "C" void dumpBool(bool b) {
   std::cout << "bool(" << (b ? "true" : "false") << ")" << std::endl;
}
extern "C" double sitofpI128F64(const __int128_t i) {
   // required for the baseline backend since TPDE currently does not support this as a snippet
   return static_cast<double>(i);
}
extern "C" float sitofpI128F32(const __int128_t i) {
   // required for the baseline backend since TPDE currently does not support this as a snippet
   return static_cast<float>(i);
}

// Runtime helpers for JIT libcalls on architectures (e.g. RISC-V) where
// compiler-rt / libgcc builtins are not automatically exported in dynamic symbol tables.
extern "C" float __floatdisf(int64_t a) {
   return static_cast<float>(a);
}
extern "C" double __floatdidf(int64_t a) {
   return static_cast<double>(a);
}
extern "C" float __floatundisf(uint64_t a) {
   return static_cast<float>(a);
}
extern "C" double __floatundidf(uint64_t a) {
   return static_cast<double>(a);
}
extern "C" float __floatsisf(int32_t a) {
   return static_cast<float>(a);
}
extern "C" double __floatsidf(int32_t a) {
   return static_cast<double>(a);
}
extern "C" float __floatunsisf(uint32_t a) {
   return static_cast<float>(a);
}
extern "C" double __floatunsidf(uint32_t a) {
   return static_cast<double>(a);
}
extern "C" int64_t __fixsfdi(float a) {
   return static_cast<int64_t>(a);
}
extern "C" int64_t __fixdfdi(double a) {
   return static_cast<int64_t>(a);
}
extern "C" uint64_t __fixunssfdi(float a) {
   return static_cast<uint64_t>(a);
}
extern "C" uint64_t __fixunsdfdi(double a) {
   return static_cast<uint64_t>(a);
}
extern "C" __int128_t __divti3(__int128_t a, __int128_t b) {
   return a / b;
}
extern "C" __int128_t __modti3(__int128_t a, __int128_t b) {
   return a % b;
}
extern "C" unsigned __int128 __udivti3(unsigned __int128 a, unsigned __int128 b) {
   return a / b;
}
extern "C" unsigned __int128 __umodti3(unsigned __int128 a, unsigned __int128 b) {
   return a % b;
}

void lingodb::execution::visitBareFunctions(const std::function<void(std::string, void*)>& fn) {
   fn("createVarLen32", reinterpret_cast<void*>(&createVarLen32));
   fn("hashVarLenData", reinterpret_cast<void*>(&hashVarLenData));
   fn("bloomMasks", reinterpret_cast<void*>(lingodb::runtime::bloomMasks));
   fn("dumpString", reinterpret_cast<void*>(&dumpString));
   fn("dumpI64", reinterpret_cast<void*>(&dumpI64));
   fn("dumpF64", reinterpret_cast<void*>(&dumpF64));
   fn("dumpBool", reinterpret_cast<void*>(&dumpBool));
   fn("sitofpI128F64", reinterpret_cast<void*>(&sitofpI128F64));
   fn("sitofpI128F32", reinterpret_cast<void*>(&sitofpI128F32));
   fn("__floatdisf", reinterpret_cast<void*>(&__floatdisf));
   fn("__floatdidf", reinterpret_cast<void*>(&__floatdidf));
   fn("__floatundisf", reinterpret_cast<void*>(&__floatundisf));
   fn("__floatundidf", reinterpret_cast<void*>(&__floatundidf));
   fn("__floatsisf", reinterpret_cast<void*>(&__floatsisf));
   fn("__floatsidf", reinterpret_cast<void*>(&__floatsidf));
   fn("__floatunsisf", reinterpret_cast<void*>(&__floatunsisf));
   fn("__floatunsidf", reinterpret_cast<void*>(&__floatunsidf));
   fn("__fixsfdi", reinterpret_cast<void*>(&__fixsfdi));
   fn("__fixdfdi", reinterpret_cast<void*>(&__fixdfdi));
   fn("__fixunssfdi", reinterpret_cast<void*>(&__fixunssfdi));
   fn("__fixunsdfdi", reinterpret_cast<void*>(&__fixunsdfdi));
   fn("__divti3", reinterpret_cast<void*>(&__divti3));
   fn("__modti3", reinterpret_cast<void*>(&__modti3));
   fn("__udivti3", reinterpret_cast<void*>(&__udivti3));
   fn("__umodti3", reinterpret_cast<void*>(&__umodti3));
}
