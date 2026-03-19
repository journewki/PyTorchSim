//  Define Instruction class, which represents single instruction inside Tile object
// Instruction object tracks dependencies, memory addresses, timing, and synchronization tags for cycle-accurate simulation

#include "Instruction.h"


// Convert an Opcode enum (MOVIN, MOVOUT, COMP, BAR) to its string name for logging/debugging
std::string opcode_to_string(Opcode opcode) {
    switch (opcode) {
        case Opcode::MOVIN:        return "MOVIN";
        case Opcode::MOVOUT:       return "MOVOUT";
        case Opcode::COMP:         return "COMP";
        case Opcode::BAR:          return "BAR";
        default:                   return "Unknown";
    }
}


/*
`start_cycle`
- Instruction execution start cycle
`finish_cycle`
- Instruction execution finish cycle
`bubble_cycle`
- Idle/bubble cycles (e.g., pipeline stalls)
`finished`
- Whether the instruction has completed
`subgraph_id`
- ID of the TileSubGraph this instruction belongs to
`_owner`
- Pointer to the owning object (typically a Tile)
`_owner_ready_queue_ref`
- Pointer to the ready queue that should be notified when this instruction becomes ready
`opcode`
- Operation type: MOVIN, MOVOUT, COMP, BAR
`compute_cycle`
- Compute cycles (from Gem5 pre-simulation) for COMP instructions
`overlapping_cycle`
- Cycles that can overlap with DMA operations (pipelining)
`ready_counter`
- Number of parent instructions that must finish before this can execute (dependency tracking)
`child_inst`
- Set of child instructions that depend on this one
`tile_size`
- Dimensions of the tile (e.g., [M, N, K] for GEMM)
`tile_stride`
- Stride per dimension for address calculation
`_tile_numel`
- Total number of elements in the tile (product of tile_size dimensions)
`_nr_waiting_request`
- Number of pending memory requests (DMA)
`_precision`
- Element size in bytes (e.g., 4 for float32, 2 for int16)
`dram_addr`
- Base DRAM address for this instruction's data
`_numa_id`
- NUMA node ID for NUMA-aware memory access
`_compute_type`
- Compute operation type (e.g., GEMM, vector op, sparse)
`_tag_idx_list`
- Tag indices for DMA synchronization/barrier matching
`_tag_stride_list`
- Strides for tag index calculation
`_tag_key`
- Computed tag key for barrier matching (set by prepare_tag_key())
`_accum_tag_idx_list`
- Accumulation dimension tag indices for reduction operations
`trace_address`
- Pre-computed address trace (e.g., for Stonne sparse operations)
`_addr_name`
- Symbolic name of the base address (e.g., "A", "B", "C" for GEMM)
`_addr_id`
- Numeric ID of the base address (for tag key generation)
`_nr_inner_loop`
- Number of inner loop iterations (for nested loop handling)
`_is_async_dma`
- Whether this DMA instruction is asynchronous (non-blocking)
`_is_indirect_mode`
- Whether this instruction uses indirect addressing (sparse access)
`_is_sparse_inst`
- Whether this is a sparse instruction (zero-skip optimization).
`_indirect_index_path`
- File path to indirect index data for sparse/indirect memory access
*/



Instruction::Instruction(Opcode opcode, cycle_type compute_cycle, size_t num_parents,
            addr_type dram_addr, std::vector<size_t> tile_size, std::vector<int> tile_stride, size_t precision,
            std::vector<int> tag_idx_list, std::vector<int> tag_stride_list,
            std::vector<int> accum_tag_idx_list)
  : opcode(opcode), compute_cycle(compute_cycle), ready_counter(num_parents), dram_addr(dram_addr),
    tile_size(tile_size), tile_stride(tile_stride), _precision(precision),
    _tag_idx_list(tag_idx_list), _tag_stride_list(tag_stride_list),
    _accum_tag_idx_list(accum_tag_idx_list) {
  assert(_tag_idx_list.size()==_tag_stride_list.size());
  _tile_numel = 1;
  for (auto dim : tile_size)
    _tile_numel *= dim;
}



// Minimal constructor: only set opcode and initializes _tile_numel to 1
Instruction::Instruction(Opcode opcode)
  : opcode(opcode) {
  _tile_numel = 1;
}



// Mark the instruction as finished, decrement ready counters of all child instructions, and set finished = true
void Instruction::finish_instruction() {
  for (auto& counter : child_inst)
    counter->dec_ready_counter();
  finished = true;
}


// Add a child instruction and increment its ready counter to track dependencies
void Instruction::add_child(std::shared_ptr<Instruction> child) {
  child->inc_ready_counter();
  child_inst.insert(child);
}


// Increment _nr_waiting_request to track pending memory requests (e.g., DMA)
void Instruction::inc_waiting_request() {
  _nr_waiting_request++;
}


// Decrement _nr_waiting_request to track pending memory requests (e.g., DMA)
void Instruction::dec_waiting_request() {
  assert(_nr_waiting_request!=0);
  _nr_waiting_request--;
}


// Build a tag key from address ID, tag indices/strides, and accumulation tag indices for DMA synchronization/barrier matching
void Instruction::prepare_tag_key() {
  /* Calculate tag key */
  int key_offset = 0;
  _tag_key.push_back(_addr_id);
  for (int i=0; i<_tag_idx_list.size(); i++)
    key_offset += _tag_idx_list.at(i) * _tag_stride_list.at(i);
  for (auto accum_dim : _accum_tag_idx_list)
    _tag_key.push_back(accum_dim);
  _tag_key.push_back(key_offset);
}


// Log the instruction's opcode as a string
void Instruction::print() {
  spdlog::info("{}", opcode_to_string(opcode));
}


// Generate the set of DRAM addresses accessed by this instruction
// Iterate over tile dimensions using strides, handle indirect addressing if enabled, and align addresses to dram_req_size boundaries
// For sparse ops (e.g. sparse attention, embedding lookup, sparse GEMM), the actual address depends on a runtime index array: address = base + stride_offset + indirect_index[i] * precision -> indirect addressing
std::shared_ptr<std::set<addr_type>> Instruction::get_dram_address(addr_type dram_req_size) {
  auto address_set = std::make_shared<std::set<addr_type>>();
  uint64_t* indirect_index = NULL;
  size_t index_count = 0;
  /* Set 4D shape*/
  while (tile_size.size() < 4)
    tile_size.insert(tile_size.begin(), 1);

  while (tile_stride.size() < 4)
    tile_stride.insert(tile_stride.begin(), 0);
  if (_is_indirect_mode) {
    spdlog::trace("[Indirect Access] Indirect mode, dump_path: {}", _indirect_index_path);
    load_indirect_index(_indirect_index_path, indirect_index, tile_size);
  }

  /* Iterate tile_size */
  for (int dim0=0; dim0<tile_size.at(0); dim0++) {
    for (int dim1=0; dim1<tile_size.at(1); dim1++) {
      for (int dim2=0; dim2<tile_size.at(2); dim2++) {
        for (int dim3=0; dim3<tile_size.at(3); dim3++) {
          addr_type address = dim0*tile_stride.at(tile_stride.size() - 4) + \
                              dim1*tile_stride.at(tile_stride.size() - 3) + \
                              dim2*tile_stride.at(tile_stride.size() - 2) + \
                              dim3*tile_stride.at(tile_stride.size() - 1);
          address = dram_addr + address * _precision;
          if (indirect_index != NULL) {
            uint64_t index_val = indirect_index[index_count++];
            address += index_val * _precision;
          }
          address_set->insert(address - (address & dram_req_size-1));
        }
      }
    }
  }
  return address_set;
}


// Load indirect index data from a binary file for sparse/indirect memory access
// Validate if file size matches expected tile element count
bool Instruction::load_indirect_index(const std::string& path, uint64_t*& indirect_index, const std::vector<uint64_t>& tile_size) {
  size_t count;
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if (!ifs) {
    spdlog::warn("[Indirect Access] Failed to open index file(\'{}\')", path);
    return false;
  }

  std::streamsize size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  count = size / sizeof(uint64_t);

  uint64_t expected_count = tile_size[0] * tile_size[1] * tile_size[2] * tile_size[3];
  if (size % sizeof(uint64_t) != 0 || count != expected_count) {
    spdlog::warn("[Indirect Access] Invalid file size ({} Bytes) at \'{}\'", size, path);
    return false;
  }

  indirect_index = new uint64_t[count];

  if (!ifs.read(reinterpret_cast<char*>(indirect_index), size)) {
    spdlog::warn("[Indirect Access] Failed to read data from file (\'{}\')", path);
    delete[] indirect_index;
    indirect_index = NULL;
    count = 0;
    return false;
  }
  return true;
}