#include "features.h"
#include "lingodb/compiler/mlir-support/eval.h"
#include "lingodb/catalog/Column.h"
#include "lingodb/catalog/Defs.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/catalog/Types.h"
#include "lingodb/execution/Execution.h"
#include "lingodb/execution/Timing.h"
#include "lingodb/runtime/Session.h"
#include "lingodb/runtime/storage/LingoDBTable.h"
#include "lingodb/scheduler/Scheduler.h"
#include "lingodb/utility/Setting.h"

#include <arrow/io/file.h>
#include <arrow/type.h>
#include <parquet/arrow/reader.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace {

lingodb::catalog::Type arrowTypeToCatalogType(const std::shared_ptr<arrow::DataType>& type) {
   using namespace lingodb::catalog;
   switch (type->id()) {
      case arrow::Type::BOOL:
         return Type::boolean();
      case arrow::Type::INT8:
         return Type::int8();
      case arrow::Type::INT16:
         return Type::int16();
      case arrow::Type::INT32:
         return Type::int32();
      case arrow::Type::INT64:
         return Type::int64();
      case arrow::Type::FLOAT:
         return Type::f32();
      case arrow::Type::DOUBLE:
         return Type::f64();
      case arrow::Type::STRING:
      case arrow::Type::LARGE_STRING:
         return Type::stringType();
      case arrow::Type::DATE32:
         return Type(LogicalTypeId::DATE, std::make_shared<DateTypeInfo>(DateTypeInfo::DateUnit::DAY));
      case arrow::Type::DATE64:
         return Type(LogicalTypeId::DATE, std::make_shared<DateTypeInfo>(DateTypeInfo::DateUnit::MILLIS));
      case arrow::Type::DECIMAL128: {
         auto decType = std::static_pointer_cast<arrow::Decimal128Type>(type);
         return Type::decimal(decType->precision(), decType->scale());
      }
      default:
         throw std::runtime_error("Unsupported arrow type in parquet: " + type->ToString());
   }
}

void registerParquetTables(lingodb::runtime::Session& session, const std::string& parquetDir) {
   if (!std::filesystem::exists(parquetDir)) {
      std::cerr << "Parquet dir does not exist: " << parquetDir << std::endl;
      return;
   }
   for (const auto& entry : std::filesystem::directory_iterator(parquetDir)) {
      if (!entry.is_regular_file()) continue;
      std::string path = entry.path().string();
      if (!path.ends_with(".parquet") || path.ends_with(".bak")) continue;

      std::string stem = entry.path().stem().string();
      auto inputFileRes = arrow::io::ReadableFile::Open(path);
      if (!inputFileRes.ok()) {
         std::cerr << "Warning: could not open " << path << ": " << inputFileRes.status().ToString() << std::endl;
         continue;
      }
      auto inputFile = inputFileRes.ValueOrDie();
      auto readerRes = parquet::arrow::OpenFile(inputFile, arrow::default_memory_pool());
      if (!readerRes.ok()) {
         std::cerr << "Warning: parquet open failed for " << path << ": " << readerRes.status().ToString() << std::endl;
         continue;
      }
      auto reader = std::move(readerRes).ValueOrDie();
      std::shared_ptr<arrow::Schema> schema;
      auto status = reader->GetSchema(&schema);
      if (!status.ok()) {
         std::cerr << "Warning: failed to read schema for " << path << std::endl;
         continue;
      }

      lingodb::catalog::CreateTableDef def;
      def.name = stem;
      for (const auto& field : schema->fields()) {
         lingodb::catalog::Type colType = arrowTypeToCatalogType(field->type());
         def.columns.emplace_back(field->name(), colType, false);
      }

      auto tableEntry = lingodb::catalog::LingoDBTableCatalogEntry::createFromCreateTable(def);
      tableEntry->setDBDir(parquetDir);
      session.getCatalog()->insertEntry(tableEntry, true);
   }
}

} // namespace

int main(int argc, char** argv) {
   using namespace lingodb;

   std::string backend = "c";
   std::string queriesDir;
   if (const char* env = std::getenv("TPCH_QUERIES_DIR")) {
      queriesDir = env;
   } else if (const char* env2 = std::getenv("QUERIES_DIR")) {
      queriesDir = env2;
   }

   std::string parquetDir;
   if (const char* env = std::getenv("PARQUET_DIR")) {
      parquetDir = env;
   } else if (const char* env2 = std::getenv("TPCH_PARQUET_DIR")) {
      parquetDir = env2;
   }
   std::vector<int> targetQueries;

   for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--help" || arg == "-h") {
         std::cout << "Usage: run-tpch-suite [--backend=<c|gpu|llvm>] [--queries=<1,3,6>] [--parquet-dir=<path>] [--queries-dir=<path>]\n"
                   << "Environment variables: TPCH_QUERIES_DIR, PARQUET_DIR (or TPCH_PARQUET_DIR)\n";
         return 0;
      } else if (arg.starts_with("--backend=")) {
         backend = arg.substr(10);
      } else if (arg == "--backend" && i + 1 < argc) {
         backend = argv[++i];
      } else if (arg.starts_with("--queries-dir=")) {
         queriesDir = arg.substr(14);
      } else if (arg == "--queries-dir" && i + 1 < argc) {
         queriesDir = argv[++i];
      } else if (arg.starts_with("--parquet-dir=")) {
         parquetDir = arg.substr(14);
      } else if (arg == "--parquet-dir" && i + 1 < argc) {
         parquetDir = argv[++i];
      } else if (arg.starts_with("--queries=")) {
         std::string qlist = arg.substr(10);
         std::stringstream ss(qlist);
         std::string item;
         while (std::getline(ss, item, ',')) {
            if (!item.empty()) targetQueries.push_back(std::stoi(item));
         }
      }
   }

   if (targetQueries.empty()) {
      for (int q = 1; q <= 22; ++q) {
         targetQueries.push_back(q);
      }
   }

   if (queriesDir.empty()) {
      std::cerr << "Error: TPC-H queries directory not specified. Use --queries-dir=<path> or set TPCH_QUERIES_DIR.\n";
      return 1;
   }
   if (parquetDir.empty()) {
      std::cerr << "Error: Parquet directory not specified. Use --parquet-dir=<path> or set PARQUET_DIR.\n";
      return 1;
   }

   compiler::support::eval::init();
   execution::ExecutionMode runMode = execution::ExecutionMode::C;
   if (backend == "c" || backend == "C") {
      runMode = execution::ExecutionMode::C;
   } else if (backend == "gpu" || backend == "GPU") {
      runMode = execution::ExecutionMode::GPU;
   } else if (backend == "llvm" || backend == "DEFAULT" || backend == "default") {
      runMode = execution::ExecutionMode::DEFAULT;
   } else if (backend == "speed" || backend == "SPEED") {
      runMode = execution::ExecutionMode::SPEED;
   }

   auto exeDir = std::filesystem::canonical(std::filesystem::path("/proc/self/exe")).parent_path();
   auto sqlTestBin = exeDir / "sql-test";
   if (!std::filesystem::exists(sqlTestBin)) {
      sqlTestBin = "sql-test";
   }

   std::cout << "========================================================\n";
   std::cout << " LingoDB TPC-H Suite Validation (ACPP PCUDA Heterogeneous)\n";
   std::cout << " Backend: " << backend << " | Parquet: " << parquetDir << "\n";
   std::cout << " Queries Dir: " << queriesDir << "\n";
   std::cout << " Driver Binary: " << sqlTestBin.string() << "\n";
   std::cout << "========================================================\n\n";

   struct QueryResult {
      bool success{false};
      double elapsedMs{0.0};
      std::string error;
   };
   std::map<int, QueryResult> results;

   for (int qNum : targetQueries) {
      std::string qPath = queriesDir + "/q" + std::to_string(qNum) + ".sql";
      if (!std::filesystem::exists(qPath)) {
         std::cerr << "[SKIP] " << qPath << " not found.\n";
         continue;
      }

      std::cout << ">>> Running TPC-H Q" << qNum << "...\n" << std::flush;
      auto start = std::chrono::high_resolution_clock::now();

      pid_t pid = fork();
      if (pid == 0) {
         std::string argBackend = "--backend=" + backend;
         std::string argParquet = "--parquet-dir=" + parquetDir;
         std::string argQuery = "--query=" + qPath;
         char* const c_args[] = {
            const_cast<char*>(sqlTestBin.c_str()),
            const_cast<char*>(argBackend.c_str()),
            const_cast<char*>(argParquet.c_str()),
            const_cast<char*>(argQuery.c_str()),
            nullptr
         };
         execv(sqlTestBin.c_str(), c_args);
         _exit(127);
      } else if (pid > 0) {
         int status = 0;
         waitpid(pid, &status, 0);
         auto end = std::chrono::high_resolution_clock::now();
         double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
         bool success = WIFEXITED(status) && (WEXITSTATUS(status) == 0);
         if (success) {
            results[qNum] = {true, elapsed, ""};
            std::cout << "[PASS] Q" << qNum << " in " << std::fixed << std::setprecision(2) << elapsed << " ms\n\n";
         } else {
            std::string reason = "Exit code " + std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1);
            if (WIFSIGNALED(status)) reason = "Killed by signal " + std::to_string(WTERMSIG(status));
            results[qNum] = {false, elapsed, reason};
            std::cout << "[FAIL] Q" << qNum << " (" << reason << ") in " << std::fixed << std::setprecision(2) << elapsed << " ms\n\n";
         }
      }
   }

   std::cout << "\n========================================================\n";
   std::cout << " TPC-H Benchmark Summary\n";
   std::cout << "========================================================\n";
   std::cout << std::left << std::setw(10) << "Query"
             << std::setw(12) << "Status"
             << std::setw(16) << "Latency (ms)"
             << "Notes\n";
   std::cout << "--------------------------------------------------------\n";
   int passCount = 0;
   for (const auto& [qNum, res] : results) {
      std::cout << std::left << std::setw(10) << ("Q" + std::to_string(qNum))
                << std::setw(12) << (res.success ? "PASS" : "FAIL")
                << std::setw(16) << std::fixed << std::setprecision(2) << res.elapsedMs
                << (res.success ? "OK" : res.error) << "\n";
      if (res.success) passCount++;
   }
   std::cout << "--------------------------------------------------------\n";
   std::cout << "Passed: " << passCount << " / " << results.size() << " queries.\n\n";

   std::quick_exit(0);
   return 0;
}

