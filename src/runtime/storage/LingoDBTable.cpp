#include "lingodb/runtime/storage/LingoDBTable.h"
#include "lingodb/catalog/Defs.h"
#include "lingodb/runtime/ArrowView.h"
#include "lingodb/runtime/storage/Restrictions.h"
#include "lingodb/scheduler/Tasks.h"
#include "lingodb/utility/Serialization.h"
#include "lingodb/utility/Tracer.h"

#include <arrow/builder.h>
#include <arrow/compute/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/table.h>
#include <parquet/arrow/reader.h>
#include "lingodb/runtime/CudaMemory.h"
#include "lingodb/runtime/GPUHashTable.hpp"


#include <algorithm>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <random>
#include <ranges>
#include <shared_mutex>
namespace {
using namespace lingodb::runtime;
namespace utility = lingodb::utility;
static utility::Tracer::Event processMorsel("TableScan", "morsel", false);

static utility::Tracer::Event processMorselSingle("TableScan", "single morsel", false);

class CudaManagedMemoryPool : public arrow::MemoryPool {
   static constexpr uint64_t MAGIC = 0xCAFEBABEDEADF00DULL;
public:
   CudaManagedMemoryPool() = default;

   arrow::Status Allocate(int64_t size, int64_t alignment, uint8_t** out) override {
      size_t actualAlignment = std::max(static_cast<size_t>(alignment), static_cast<size_t>(64));
      size_t actualSize = size + actualAlignment;
      void* ptr = cudaAllocManaged(actualSize);
      if (ptr) {
         uint64_t* header = reinterpret_cast<uint64_t*>(ptr);
         header[0] = MAGIC;
         header[1] = actualAlignment;
         *out = reinterpret_cast<uint8_t*>(ptr) + actualAlignment;
         return arrow::Status::OK();
      }
      return arrow::default_memory_pool()->Allocate(size, alignment, out);
   }

   arrow::Status Reallocate(int64_t old_size, int64_t new_size, int64_t alignment, uint8_t** ptr) override {
      uint8_t* new_ptr = nullptr;
      auto st = Allocate(new_size, alignment, &new_ptr);
      if (!st.ok()) return st;
      if (*ptr && old_size > 0) {
         memcpy(new_ptr, *ptr, std::min(old_size, new_size));
      }
      Free(*ptr, old_size, alignment);
      *ptr = new_ptr;
      return arrow::Status::OK();
   }

   void Free(uint8_t* buffer, int64_t size, int64_t alignment) override {
      if (!buffer) return;
      size_t actualAlignment = std::max(static_cast<size_t>(alignment), static_cast<size_t>(64));
      void* originalPtr = buffer - actualAlignment;
      uint64_t* header = reinterpret_cast<uint64_t*>(originalPtr);
      if (header[0] == MAGIC) {
         header[0] = 0;
         cudaFreeManaged(originalPtr);
         return;
      }
      arrow::default_memory_pool()->Free(buffer, size, alignment);
   }

   int64_t bytes_allocated() const override { return 0; }
   int64_t max_memory() const override { return 0; }
   int64_t total_bytes_allocated() const override { return 0; }
   int64_t num_allocations() const override { return 0; }
   std::string backend_name() const override { return "cuda_managed"; }
};

static arrow::MemoryPool* getCudaManagedMemoryPool() {
   static CudaManagedMemoryPool pool;
   return &pool;
}

std::vector<lingodb::runtime::LingoDBTable::TableChunk> loadTable(std::string name) {
   if (name.ends_with(".parquet")) {
      auto inputFileRes = arrow::io::ReadableFile::Open(name);
      if (!inputFileRes.ok()) {
         throw std::runtime_error("Could not open parquet file: " + name + " (" + inputFileRes.status().ToString() + ")");
      }
      auto inputFile = inputFileRes.ValueOrDie();
      auto readerRes = parquet::arrow::OpenFile(inputFile, getCudaManagedMemoryPool());
      if (!readerRes.ok()) {
         throw std::runtime_error("Could not open parquet reader: " + readerRes.status().ToString());
      }
      auto reader = std::move(readerRes).ValueOrDie();
      reader->set_use_threads(true);

      std::shared_ptr<arrow::Table> table;
      auto st = reader->ReadTable(&table);
      if (!st.ok() || !table) {
         throw std::runtime_error("Failed to read parquet table: " + st.ToString());
      }

      std::vector<lingodb::runtime::LingoDBTable::TableChunk> chunks;
      arrow::TableBatchReader batchReader(*table);
      batchReader.set_chunksize(65536);
      std::shared_ptr<arrow::RecordBatch> batch;
      size_t currRowId = 0;
      while (batchReader.ReadNext(&batch).ok() && batch) {
         chunks.emplace_back(batch, currRowId);
         currRowId += batch->num_rows();
      }
      return chunks;
   }
   auto inputFile = arrow::io::ReadableFile::Open(name).ValueOrDie();
   auto batchReader = arrow::ipc::RecordBatchFileReader::Open(inputFile).ValueOrDie();
   std::vector<lingodb::runtime::LingoDBTable::TableChunk> batches;
   size_t currRowId = 0;
   for (int i = 0; i < batchReader->num_record_batches(); i++) {
      auto batch = batchReader->ReadRecordBatch(i).ValueOrDie();
      batches.push_back(lingodb::runtime::LingoDBTable::TableChunk(batch, currRowId));
      currRowId += batch->num_rows();
   }
   return batches;
}
void storeTable(std::string file, std::shared_ptr<arrow::Schema> schema, const std::vector<lingodb::runtime::LingoDBTable::TableChunk>& data) {
   auto inputFile = arrow::io::FileOutputStream::Open(file).ValueOrDie();
   auto batchWriter = arrow::ipc::MakeFileWriter(inputFile, schema).ValueOrDie();
   for (auto& batch : data) {
      if (!batchWriter->WriteRecordBatch(*batch.data()).ok()) {
         throw std::runtime_error("could not store table");
      }
   }
   auto close1 = batchWriter->Close();
   if (!close1.ok()) {
      throw std::runtime_error("could not store table:" + close1.ToString());
   }
   if (!inputFile->Close().ok()) {
      throw std::runtime_error("could not store table");
   }
}
/*
 * Create sample from arrow table
 */

template <typename I>
class BoxedIntegerIterator {
   I i;

   public:
   typedef I difference_type;
   typedef I value_type;
   typedef I pointer;
   typedef I reference;
   typedef std::random_access_iterator_tag iterator_category;

   BoxedIntegerIterator(I i) : i{i} {}

   bool operator==(BoxedIntegerIterator<I>& other) { return i == other.i; }
   I operator-(BoxedIntegerIterator<I>& other) { return i - other.i; }
   I operator++() { return i++; }
   I operator*() { return i; }
};
std::shared_ptr<arrow::RecordBatch> createSample(const std::vector<lingodb::runtime::LingoDBTable::TableChunk>& data) {
   size_t numRows = 0;
   for (auto& batch : data) {
      numRows += batch.data()->num_rows();
   }
   if (numRows == 0) {
      return std::shared_ptr<arrow::RecordBatch>();
   }
   std::vector<size_t> result;
   size_t sampleCount = std::min<size_t>(numRows, 1024ull);
   result.reserve(sampleCount);
   auto rng = std::mt19937{std::random_device{}()};
   std::uniform_int_distribution<size_t> dist(0, numRows - 1);
   std::unordered_set<size_t> picked;
   while (result.size() < sampleCount) {
      size_t idx = dist(rng);
      if (picked.insert(idx).second) {
         result.push_back(idx);
      }
   }
   std::sort(result.begin(), result.end());
   size_t currPos = 0;
   size_t currBatch = 0;
   size_t batchStart = 0;
   std::vector<std::shared_ptr<arrow::RecordBatch>> sampleData;
   while (currPos < result.size()) {
      std::vector<size_t> fromCurrentBatch;
      while (currPos < result.size() && result[currPos] < batchStart + data[currBatch].data()->num_rows()) {
         fromCurrentBatch.push_back(result[currPos] - batchStart);
         currPos++;
      }
      if (fromCurrentBatch.size() > 0) {
         arrow::NumericBuilder<arrow::Int32Type> numericBuilder;
         for (auto i : fromCurrentBatch) {
            if (!numericBuilder.Append(i).ok()) {
               throw std::runtime_error("could not create sample");
            }
         }
         auto indices = numericBuilder.Finish().ValueOrDie();
         std::vector<arrow::Datum> args({data[currBatch].data(), indices});
         auto res = arrow::compute::CallFunction("take", args).ValueOrDie();
         sampleData.push_back(res.record_batch());
      }
      batchStart += data[currBatch].data()->num_rows();
      currBatch++;
   }
   return arrow::Table::FromRecordBatches(sampleData).ValueOrDie()->CombineChunksToBatch().ValueOrDie();
}

std::shared_ptr<arrow::DataType> toPhysicalType(lingodb::catalog::Type t) {
   using TypeId = lingodb::catalog::LogicalTypeId;
   switch (t.getTypeId()) {
      case TypeId::BOOLEAN:
         return arrow::boolean();
      case TypeId::INT:
         switch (t.getInfo<lingodb::catalog::IntTypeInfo>()->getBitWidth()) {
            case 8:
               return arrow::int8();
            case 16:
               return arrow::int16();
            case 32:
               return arrow::int32();
            case 64:
               return arrow::int64();
            default:
               throw std::runtime_error("unsupported bit width");
         }
      case TypeId::FLOAT:
         return arrow::float32();
      case TypeId::DOUBLE:
         return arrow::float64();
      case TypeId::DECIMAL:
         return arrow::decimal128(t.getInfo<lingodb::catalog::DecimalTypeInfo>()->getPrecision(),
                                  t.getInfo<lingodb::catalog::DecimalTypeInfo>()->getScale());
      case TypeId::DATE: {
         auto dateUnit = t.getInfo<lingodb::catalog::DateTypeInfo>()->getUnit();
         switch (dateUnit) {
            case lingodb::catalog::DateTypeInfo::DateUnit::DAY:
               return arrow::date32();
            case lingodb::catalog::DateTypeInfo::DateUnit::MILLIS:
               return arrow::date64();
         }
      }
      case TypeId::TIMESTAMP: {
         arrow::TimeUnit::type timeUnit;
         auto logicalTimeUnit = t.getInfo<lingodb::catalog::TimestampTypeInfo>()->getUnit();
         switch (logicalTimeUnit) {
            case lingodb::catalog::TimestampTypeInfo::TimestampUnit::NANOS:
               timeUnit = arrow::TimeUnit::NANO;
               break;
            case lingodb::catalog::TimestampTypeInfo::TimestampUnit::MICROS:
               timeUnit = arrow::TimeUnit::MICRO;
               break;
            case lingodb::catalog::TimestampTypeInfo::TimestampUnit::MILLIS:
               timeUnit = arrow::TimeUnit::MILLI;
               break;
            case lingodb::catalog::TimestampTypeInfo::TimestampUnit::SECONDS:
               timeUnit = arrow::TimeUnit::SECOND;
               break;
         }
         return arrow::timestamp(timeUnit);
      }
      case TypeId::INTERVAL: {
         auto intervalUnit = t.getInfo<lingodb::catalog::IntervalTypeInfo>()->getUnit();
         switch (intervalUnit) {
            case lingodb::catalog::IntervalTypeInfo::IntervalUnit::DAYTIME:
               return arrow::day_time_interval();
            case lingodb::catalog::IntervalTypeInfo::IntervalUnit::MONTH:
               return arrow::month_interval();
         }
      }
      case TypeId::CHAR: {
         if (t.getInfo<lingodb::catalog::CharTypeInfo>()->getLength() == 1) {
            return arrow::fixed_size_binary(4);
         }
         return arrow::utf8();
      }
      case TypeId::STRING:
         return arrow::utf8();
      default:
         throw std::runtime_error("unsupported type");
   }
}

} // namespace

namespace lingodb::runtime {
LingoDBTable::TableChunk::TableChunk(std::shared_ptr<arrow::RecordBatch> data, size_t startRowId) : internalData(data), startRowId(startRowId), numRows(data->num_rows()) {
   std::vector<size_t> bufferStart;
   for (auto colId = 0; colId < data->num_columns(); colId++) {
      auto arrayData = data->column(colId)->data();
      size_t currBufId = buffers.size();
      for (size_t i = 0; i < arrayData->buffers.size(); i++) {
         auto buffer = arrayData->buffers[i];
         if (buffer) {
            buffers.push_back(buffer->data());
         } else {
            buffers.push_back(nullptr);
         }
      }
      if (!buffers[currBufId]) {
         if (numRows > ArrayView::maxNullCount) {
            throw std::runtime_error("LingoDBTable: too many nulls in column");
         }
         buffers[currBufId] = ArrayView::validData.data();
      }
      bufferStart.push_back(currBufId);
   }
   for (auto colId = 0; colId < data->num_columns(); colId++) {
      auto arrayData = data->column(colId)->data();
      columnInfo.push_back(ArrayView{.length = arrayData->length, .nullCount = arrayData->null_count, .offset = arrayData->offset, .nBuffers = static_cast<int64_t>(arrayData->buffers.size()), .nChildren = static_cast<int64_t>(arrayData->child_data.size()), .buffers = &buffers[bufferStart.at(colId)], .children = nullptr});
   }
}

std::unique_ptr<LingoDBTable> LingoDBTable::create(const catalog::CreateTableDef& def) {
   arrow::FieldVector fields;
   for (auto c : def.columns) {
      fields.push_back(std::make_shared<arrow::Field>(std::string{c.getColumnName()}, toPhysicalType(c.getLogicalType())));
   }
   auto arrowSchema = std::make_shared<arrow::Schema>(fields);
   return std::make_unique<LingoDBTable>(def.name + ".arrow", arrowSchema);
}
LingoDBTable::LingoDBTable(std::string fileName, std::shared_ptr<arrow::Schema> arrowSchema) : persist(false), fileName(std::move(fileName)), sample(arrowSchema), schema(std::move(arrowSchema)), tableData(), numRows(0) {
   for (auto c : schema->fields()) {
      columnStatistics[c->name()] = catalog::ColumnStatistics(utility::HyperLogLog());
   }
}
void LingoDBTable::append(const std::shared_ptr<arrow::Table>& table) {
   std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
   arrow::TableBatchReader reader(table);
   std::shared_ptr<arrow::RecordBatch> nextChunk;
   while (reader.ReadNext(&nextChunk) == arrow::Status::OK()) {
      if (nextChunk) {
         batches.push_back(nextChunk);
      } else {
         break;
      }
      nextChunk.reset();
   }
   append(batches);
}
void LingoDBTable::append(const std::vector<std::shared_ptr<arrow::RecordBatch>>& toAppend) {
   ensureLoaded();
   for (auto& batch : toAppend) {
      if (batch->schema()->Equals(*schema)) {
         tableData.push_back(TableChunk{batch, numRows});
         numRows += batch->num_rows();
      } else {
         std::cout << "schema to add: " << batch->schema()->ToString() << std::endl;
         std::cout << "schema of table: " << schema->ToString() << std::endl;
         throw std::runtime_error("schema mismatch");
      }
      for (auto c : schema->fields()) {
         columnStatistics[c->name()].merge(batch->GetColumnByName(c->name()));
      }
   }
   sample = createSample(tableData);
   flush();
}
const catalog::ColumnStatistics& LingoDBTable::getColumnStatistics(std::string_view column) const {
   if (!columnStatistics.contains(column)) {
      throw std::runtime_error("MetaData: Column not found");
   }
   // the "at" is only available heterogeneously from C++26 onwards
   // https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2363r5.html
   return columnStatistics.at(std::string{column});
}
void LingoDBTable::flush() {
   if (!persist) return;
   ensureLoaded();
   storeTable(dbDir + "/" + fileName, schema, tableData);
}

std::shared_ptr<arrow::DataType> LingoDBTable::getColumnStorageType(std::string_view columnName) const {
   // this unnecessary to-string-allocation must be fixed in arrow. for now, we use this workaround
   auto field = schema->GetFieldByName(std::string{columnName});
   if (!field) {
      throw std::runtime_error("column not found");
   }
   return field->type();
}
void LingoDBTable::ensureLoaded() {
   if (!loaded) {
      loaded = true;
      if (fileName.empty() && dbDir.empty()) {
         return;
      }
      auto fsPath = dbDir.empty() ? std::filesystem::path(fileName) : (std::filesystem::path(dbDir) / fileName);
      if (!std::filesystem::exists(fsPath)) {
         if (fileName.ends_with(".arrow")) {
            auto alt = fsPath;
            alt.replace_extension(".parquet");
            if (std::filesystem::exists(alt)) {
               fsPath = alt;
            }
         }
      }
      if (!std::filesystem::exists(fsPath)) {
         return;
      }
      tableData = loadTable(fsPath.string());
      if (numRows == 0) {
         for (const auto& chunk : tableData) {
            numRows += chunk.getNumRows();
         }
      }
      if (numRows > 0 && (!sample.getSampleData() || sample.getSampleData()->num_rows() == 0)) {
         sample = createSample(tableData);
      }
   }
}
GPUHashTable* LingoDBTable::getOrCreateGPUHashTable(size_t colIndex) {
   ensureLoaded();
   auto it = gpuHashTables.find(colIndex);
   if (it != gpuHashTables.end()) {
      return it->second.get();
   }
   auto ht = std::shared_ptr<GPUHashTable>(GPUHashTable::create(numRows), GPUHashTable::destroy);
   int64_t rowOffset = 0;
   auto typeId = schema->field(colIndex)->type()->id();
   for (const auto& chunk : tableData) {
      const auto* arrView = chunk.getArrayView(colIndex);
      if (arrView && arrView->buffers && arrView->nBuffers > 1 && arrView->buffers[1]) {
         if (typeId == arrow::Type::INT32 || typeId == arrow::Type::DATE32) {
            const int32_t* colData = reinterpret_cast<const int32_t*>(arrView->buffers[1]);
            for (size_t r = 0; r < chunk.getNumRows(); ++r) {
               ht->insert(static_cast<int64_t>(colData[r]), rowOffset + static_cast<int64_t>(r));
            }
         } else if (typeId == arrow::Type::INT64) {
            const int64_t* colData = reinterpret_cast<const int64_t*>(arrView->buffers[1]);
            for (size_t r = 0; r < chunk.getNumRows(); ++r) {
               ht->insert(colData[r], rowOffset + static_cast<int64_t>(r));
            }
         } else {
            const int32_t* colData = reinterpret_cast<const int32_t*>(arrView->buffers[1]);
            for (size_t r = 0; r < chunk.getNumRows(); ++r) {
               ht->insert(static_cast<int64_t>(colData[r]), rowOffset + static_cast<int64_t>(r));
            }
         }
      }
      rowOffset += static_cast<int64_t>(chunk.getNumRows());
   }
   auto* ptr = ht.get();
   gpuHashTables[colIndex] = std::move(ht);
   return ptr;
}
void LingoDBTable::serialize(lingodb::utility::Serializer& serializer) const {
   serializer.writeProperty(1, fileName);
   serializer.writeProperty(2, sample);
   auto res = arrow::ipc::SerializeSchema(*schema).ValueOrDie();
   serializer.writeProperty(3, std::string_view((const char*) res->data(), res->size()));
   serializer.writeProperty(4, columnStatistics);
   serializer.writeProperty(5, numRows);
}

class BatchesWorkerResvState {
   public:
   size_t batchId;
   std::shared_mutex mutex;
   size_t resvCursor{0};
   size_t resvId{0};
   size_t unitAmount;
   // workerId steal task from
   size_t stealWorkerId{std::numeric_limits<size_t>::max()};

   bool hasMoreWork() {
      std::shared_lock<std::shared_mutex> stateLock(this->mutex);
      return resvCursor < unitAmount;
   }
   int fetchAndNextOwn(size_t splitSize, std::atomic<size_t>& startIndex, const std::vector<lingodb::runtime::LingoDBTable::TableChunk>& batches) {
      std::unique_lock<std::shared_mutex> stateLock(this->mutex);
      size_t curr = resvCursor++;

      if (curr < unitAmount) {
         return curr;
      }
      //2. if the current worker has no more work locally, try to allocate new work
      size_t localStartIndex = startIndex.fetch_add(1);
      if (localStartIndex < batches.size()) {
         auto& buffer = batches[localStartIndex];
         unitAmount = (buffer.getNumRows() + splitSize - 1) / splitSize;
         resvCursor = 1;
         resvId = 0;
         batchId = localStartIndex;

         return 0;
      }
      return -1;
   }
   int fetchAndNext() {
      std::unique_lock<std::shared_mutex> resvLock(mutex);
      size_t cur = resvCursor++;
      return cur >= unitAmount ? -1 : cur;
   }
};

class ScanBatchesTask : public lingodb::scheduler::TaskWithImplicitContext {
   std::vector<lingodb::runtime::LingoDBTable::TableChunk>& batches;
   std::vector<size_t> colIds;
   std::unique_ptr<Restrictions> restrictions;
   std::function<void(lingodb::runtime::BatchView*)> cb;
   std::vector<lingodb::runtime::BatchView> batchInfos;
   std::vector<std::vector<const ArrayView*>> arrayViewPtrs;
   std::atomic<size_t> startIndex{0};
   size_t splitSize{20000};
   std::vector<std::unique_ptr<BatchesWorkerResvState>> workerResvs;
   std::vector<std::pair<uint16_t*, uint16_t*>> selVecs;

   public:
   ScanBatchesTask(LingoDBTable& table, std::vector<lingodb::runtime::LingoDBTable::TableChunk>& batches, std::vector<size_t> colIds, std::unique_ptr<Restrictions> restrictions, const std::function<void(lingodb::runtime::BatchView*)>& cb) : batches(batches), colIds(colIds), restrictions(std::move(restrictions)), cb(cb) {
      size_t smallerSplitSize = std::max(1000ul, table.getNumRows() / (lingodb::scheduler::getNumWorkers() * 2));
      if (smallerSplitSize < splitSize) {
         splitSize = smallerSplitSize;
      }
      for (size_t i = 0; i < lingodb::scheduler::getNumWorkers(); i++) {
         batchInfos.emplace_back(lingodb::runtime::BatchView());
         arrayViewPtrs.emplace_back(std::vector<const ArrayView*>(colIds.size()));
         batchInfos[i].arrays = arrayViewPtrs[i].data();
         auto* s1 = reinterpret_cast<uint16_t*>(cudaAllocManaged(splitSize * sizeof(uint16_t)));
         auto* s2 = reinterpret_cast<uint16_t*>(cudaAllocManaged(splitSize * sizeof(uint16_t)));
         selVecs.emplace_back(std::make_pair(s1, s2));
         workerResvs.emplace_back(std::make_unique<BatchesWorkerResvState>());
      }
   }
   void unitRun(size_t batchId, int unitId) {
      auto& chunk = batches[batchId];
      if (unitId < 0) {
         return;
      }
      auto [selVec1, selVec2] = selVecs[lingodb::scheduler::currentWorkerId()];
      size_t begin = splitSize * unitId;
      size_t len = std::min(begin + splitSize, chunk.getNumRows()) - begin;
      BatchView& batchView = batchInfos[lingodb::scheduler::currentWorkerId()];
      batchView.offset = begin;
      batchView.selectionVector = BatchView::defaultSelectionVector.data();
      batchView.length = std::min(static_cast<size_t>(chunk.getNumRows() - begin), len);

      for (size_t i = 0; i < colIds.size(); i++) {
         batchView.arrays[i] = chunk.getArrayView(colIds[i]);
      }
      auto [newLen, selVec] = restrictions->applyFilters(begin, batchView.length, selVec1, selVec2, [&](size_t colId) {
         return chunk.getArrayView(colId);
      });
      batchView.length = newLen;
      batchView.selectionVector = selVec;
      if (batchView.length > 0) {
         utility::Tracer::Trace trace(processMorsel);
         cb(&batchView);
      }
   }

   bool allocateWork() override {
      // quick check for exhaust. workExhausted is true if there is no more buffer or no more
      // work unit in own local state or steal from other workers.
      if (workExhausted.load()) {
         return false;
      }

      //1. if the current worker has more work locally, do it
      auto* state = workerResvs[lingodb::scheduler::currentWorkerId()].get();
      auto id = state->fetchAndNextOwn(splitSize, startIndex, batches);
      if (id != -1) {
         state->resvId = id;
         return true;
      }

      //3. if the current worker has no more work locally and no more work globally, try to steal work from the worker we stole from last time
      if (state->stealWorkerId != std::numeric_limits<size_t>::max()) {
         auto* other = workerResvs[state->stealWorkerId].get();
         if (other->hasMoreWork()) {
            auto id = other->fetchAndNext();
            if (id != -1) {
               state->resvId = id;
               return true;
            }
         }
         state->stealWorkerId = std::numeric_limits<size_t>::max();
      }
      //4. if the current worker has no more work locally and no more work globally, try to steal work from other workers
      for (size_t i = 1; i < workerResvs.size(); i++) {
         // make sure index of worker to steal never exceed worker number limits
         auto idx = (lingodb::scheduler::currentWorkerId() + i) % workerResvs.size();
         auto* other = workerResvs[idx].get();
         if (other->hasMoreWork()) {
            auto id = other->fetchAndNext();
            if (id != -1) {
               // only current worker can modify its onw stealWorkerId. no need to lock
               state->stealWorkerId = idx;
               state->resvId = id;
               return true;
            }
         }
      }

      workExhausted.store(true);
      return false;
   }
   void performWork() override {
      auto* state = workerResvs[lingodb::scheduler::currentWorkerId()].get();
      if (state->stealWorkerId != std::numeric_limits<size_t>::max()) {
         auto* other = workerResvs[state->stealWorkerId].get();
         unitRun(other->batchId, state->resvId);
         return;
      }
      unitRun(state->batchId, state->resvId);
   }
   ~ScanBatchesTask() {
      for (auto& selVec : selVecs) {
         cudaFreeManaged(selVec.first);
         cudaFreeManaged(selVec.second);
      }
   }
};
std::unique_ptr<LingoDBTable> LingoDBTable::deserialize(lingodb::utility::Deserializer& deserializer) {
   auto fileName = deserializer.readProperty<std::string>(1);
   auto sample = deserializer.readProperty<catalog::Sample>(2);
   auto schemaData = deserializer.readProperty<std::string>(3);
   arrow::ipc::DictionaryMemo dictMemo;
   auto bufferReader = arrow::io::BufferReader::FromString(schemaData);
   auto schema = arrow::ipc::ReadSchema(bufferReader.get(), &dictMemo).ValueOrDie();
   auto columnStatistics = deserializer.readProperty<LingoDBTable::ColumnStatisticsMap>(4);
   auto numRows = deserializer.readProperty<size_t>(5);
   return std::make_unique<LingoDBTable>(fileName, schema, numRows, sample, columnStatistics);
}

class ScanBatchesSingleThreadedTask : public lingodb::scheduler::TaskWithImplicitContext {
   std::vector<lingodb::runtime::LingoDBTable::TableChunk>& batches;
   std::vector<size_t> colIds;
   std::unique_ptr<Restrictions> restrictions;
   std::function<void(lingodb::runtime::BatchView*)> cb;

   public:
   ScanBatchesSingleThreadedTask(std::vector<lingodb::runtime::LingoDBTable::TableChunk>& batches, std::vector<size_t> colIds, std::unique_ptr<Restrictions> restrictions, const std::function<void(lingodb::runtime::BatchView*)>& cb) : batches(batches), colIds(colIds), restrictions(std::move(restrictions)), cb(cb) {
   }

   bool allocateWork() override {
      if (!workExhausted.exchange(true)) {
         return true;
      }
      return false;
   }
   void performWork() override {
      BatchView batchView;
      std::vector<const ArrayView*> arrayViewPtrs(colIds.size());
      batchView.arrays = arrayViewPtrs.data();
      batchView.offset = 0;
      batchView.length = 0;
      uint16_t* selVec1 = reinterpret_cast<uint16_t*>(cudaAllocManaged(BatchView::maxBatchSize * sizeof(uint16_t)));
      uint16_t* selVec2 = reinterpret_cast<uint16_t*>(cudaAllocManaged(BatchView::maxBatchSize * sizeof(uint16_t)));

      for (auto& batch : batches) {
         utility::Tracer::Trace trace(processMorselSingle);
         for (size_t start = 0; start < batch.getNumRows(); start += BatchView::maxBatchSize) {
            size_t len = batch.getNumRows() - start;
            batchView.offset = start;
            batchView.length = len;
            for (size_t i = 0; i < colIds.size(); i++) {
               batchView.arrays[i] = batch.getArrayView(colIds[i]);
            }
            // apply restrictions
            auto [newLen, selVec] = restrictions->applyFilters(start, len, selVec1, selVec2, [&](size_t colId) { return batch.getArrayView(colId); });
            batchView.length = newLen;
            batchView.selectionVector = selVec;
            if (batchView.length > 0) {
               cb(&batchView);
            }
         }
         trace.stop();
      }
      cudaFreeManaged(selVec1);
      cudaFreeManaged(selVec2);
   }
   ~ScanBatchesSingleThreadedTask() {
   }
};

std::unique_ptr<scheduler::Task> LingoDBTable::createScanTask(const ScanConfig& scanConfig) {
   ensureLoaded();
   std::vector<size_t> colIds;
   for (const auto& c : scanConfig.columns) {
      auto colId = schema->GetFieldIndex(c);
      assert(colId >= 0);
      colIds.push_back(colId);
   }
   auto restrictions = lingodb::runtime::Restrictions::create(scanConfig.filters, *schema);
   if (scanConfig.parallel) {
      return std::make_unique<ScanBatchesTask>(*this, tableData, colIds, std::move(restrictions), scanConfig.cb);
   } else {
      return std::make_unique<ScanBatchesSingleThreadedTask>(tableData, colIds, std::move(restrictions), scanConfig.cb);
   }
}

std::pair<const LingoDBTable::TableChunk*, size_t> LingoDBTable::getByRowId(size_t rowId) const {
   auto res = std::upper_bound(tableData.begin(), tableData.end(), rowId, [](size_t rowId, const TableChunk& chunk) { return rowId < chunk.startRowId + chunk.numRows; });
   if (res == tableData.end()) {
      throw std::runtime_error("row id out of bounds");
   }
   auto& chunk = *res;
   return {&chunk, rowId - chunk.startRowId};
}

size_t LingoDBTable::getColIndex(std::string colName) {
   return schema->GetFieldIndex(colName);
}

} // namespace lingodb::runtime
