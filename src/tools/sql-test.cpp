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

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

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
      // e.g. "lineitem"
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
   std::string queryStr;
   std::string queryFile;
   std::string parquetDir;
   if (const char* env = std::getenv("PARQUET_DIR")) {
      parquetDir = env;
   } else if (const char* env2 = std::getenv("TPCH_PARQUET_DIR")) {
      parquetDir = env2;
   }
   std::string dbDir;

   for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--help" || arg == "-h") {
         std::cout << "Usage: sql-test [--backend=<c|gpu|llvm>] [--query=<file>] [--parquet-dir=<path>] [--db-dir=<path>]\n"
                   << "Environment variables: PARQUET_DIR (or TPCH_PARQUET_DIR), ACPP_BIN, ACPP_DIR\n";
         return 0;
      } else if (arg.starts_with("--backend=")) {
         backend = arg.substr(10);
      } else if (arg == "--backend" && i + 1 < argc) {
         backend = argv[++i];
      } else if (arg.starts_with("--query=")) {
         queryFile = arg.substr(8);
      } else if (arg == "--query" && i + 1 < argc) {
         queryFile = argv[++i];
      } else if (arg.starts_with("--parquet-dir=")) {
         parquetDir = arg.substr(14);
      } else if (arg == "--parquet-dir" && i + 1 < argc) {
         parquetDir = argv[++i];
      } else if (arg.starts_with("--db-dir=")) {
         dbDir = arg.substr(9);
      } else if (arg == "--db-dir" && i + 1 < argc) {
         dbDir = argv[++i];
      } else if (!arg.starts_with("--")) {
         if (queryFile.empty()) queryFile = arg;
         else if (dbDir.empty()) dbDir = arg;
      }
   }

   if (queryFile.empty() && queryStr.empty()) {
      std::cerr << "Usage: sql-test --backend=c [--parquet-dir <path>] --query <query.sql>" << std::endl;
      return 1;
   }
   if (parquetDir.empty() && dbDir.empty()) {
      std::cerr << "Error: Neither --parquet-dir (or PARQUET_DIR) nor --db-dir was specified.\n";
      return 1;
   }

   // Read query from file if file exists
   if (!queryFile.empty() && std::filesystem::exists(queryFile)) {
      std::ifstream f(queryFile);
      std::stringstream buffer;
      buffer << f.rdbuf();
      queryStr = buffer.str();
   } else if (!queryFile.empty()) {
      queryStr = queryFile;
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

   auto scheduler = scheduler::startScheduler();
   std::shared_ptr<runtime::Session> session;
   if (!parquetDir.empty() && std::filesystem::exists(parquetDir)) {
      session = runtime::Session::createSession(parquetDir, false);
      registerParquetTables(*session, parquetDir);
   } else if (!dbDir.empty()) {
      session = runtime::Session::createSession(dbDir, false);
   } else {
      session = runtime::Session::createSession();
   }

   auto queryExecutionConfig = execution::createQueryExecutionConfig(runMode, true);
   auto executer = execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig), *session);
   executer->fromData(queryStr);
   scheduler::awaitEntryTask(std::make_unique<execution::QueryExecutionTask>(std::move(executer)));

   std::quick_exit(0);
   return 0;
}
