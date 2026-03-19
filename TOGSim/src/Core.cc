// Define Core class which models one NPU core. One core is a compute unit responsible for issuing and executing instructions, access memory by DMA or through NoC. 
// Core has vector unit and systolic array inside and each instruction is sent to corresponding pipeline by compute type. 
// It holds in‑flight tiles, runs vector/systolic compute pipelines, drives the DMA engine, and interfaces with the interconnect/DRAM, while collecting per‑core stats.


/*
Attributes
<Identity & config>
`_id`
- Core id
`_config`
- SimulationConfig for this core
`_num_systolic_array_per_core`
- Number of systolic arrays attached to this core
`_systolic_array_rr`
- Round‑robin index for assigning compute to systolic arrays

<DMA & memory interface>
`_dma`
- DMA engine for this core
- Generate memory request and send to request queue of each core
`_request_queue`
- Outgoing mem_fetch* requests to the interconnect
`_response_queue`
- Responses handled via push_memory_response
`_dma_waiting_queue` 
- Map from Instruction to shared_ptr for DMA instructions waiting on memory responses
`_dma_finished_queue`
- List of DMA instructions that have completed and need post‑processing
`_waiting_write_reqs`
- Track outstanding write requests

<Cycle counters & statistics>
`_core_cycle`
- Current cycle count of the core
`_stat_tot_vu_compute_cycle`, _`stat_vu_compute_cycle`
- Total / window vector‑unit active cycles
`_stat_tot_sa_compute_cycle`, `_stat_sa_compute_cycle`
- Per‑SA total / window active cycle
`_stat_tot_dma_cycle`, `_stat_dma_cycle`
- Total / window DMA active cycles.
`_stat_tot_dma_idle_cycle`, `_stat_dma_idle_cycle`
- Total / window DMA idle cycles
`_stat_tot_vu_compute_idle_cycle`, `_stat_vu_compute_idle_cycle`
- Total / window VU idle cycles
`_stat_tot_sa_compute_idle_cycle`, `_stat_sa_compute_idle_cycle`
- Per‑SA total / window idle cycles
`_stat_inst_count`
- Per‑opcode issued instruction counts
`_stat_tot_skipped_inst`
- Per‑opcode skipped instruction counts (e.g., sparsity reuse)
`_stat_tot_mem_response`, `_stat_mem_response`
- Total / window memory responses seen
`_stat_gemm_inst`
- Number of GEMM‑type compute instructions
`_stat_skip_dma
- DMA skips (e.g., sparse tiles)
`_stat_numa_local_access`, `_stat_numa_remote_access`
- NUMA locality counters

<Tiles and pipelines>
`_tiles`
- List of currently active Tile objects on this core
`_finished_tiles`
- Queue of completed tiles ready to be popped by Simulator
`_vu_compute_pipeline`
- Queue of compute instructions for the vector unit
`_sa_compute_pipeline`
- Vector of queues, one per systolic array
`_ld_inst_queue`, `_st_inst_queue`
- queues of load/store instructions pending DMA issue
*/



#include "Core.h"

// <Related to lifecycle/scheduling> 
// Initialize DMA, pipelines, and stat vectors etc of Core object
Core::Core(uint32_t id, SimulationConfig config)
    : _id(id),
      _config(config),
      _core_cycle(0),
      _stat_dma_cycle(0),
      _num_systolic_array_per_core(config.num_systolic_array_per_core),
      _dma(id, config.dram_req_size) {
  _sa_compute_pipeline.resize(_num_systolic_array_per_core);
  _stat_tot_sa_compute_cycle.resize(_num_systolic_array_per_core);
  _stat_sa_compute_cycle.resize(_num_systolic_array_per_core);
  _stat_tot_sa_compute_idle_cycle.resize(_num_systolic_array_per_core);
  _stat_sa_compute_idle_cycle.resize(_num_systolic_array_per_core);
  _stat_inst_count.resize(static_cast<size_t>(Opcode::COUNT), 0);
  _stat_tot_skipped_inst.resize(static_cast<size_t>(Opcode::COUNT), 0);
}

// Check if this core can accept another tile (e.g. limit on concurrent tiles, skip Stonne tiles)
// One Core can hold up to 4 tiles at most(doesn’t check SRAM exactly currently)
bool Core::can_issue(const std::shared_ptr<Tile>& op) {
  /* Check SRAM is enough to run tile */
  return _tiles.size() < 4  && !op->is_stonne_tile();
}


// Accept a tile(add the tile to _tiles of the core), enqueue all ready instructions in the tile
void Core::issue(std::shared_ptr<Tile> op) {
  if (op->get_instructions().size()){
    spdlog::trace("[{}][Core {}][TILE_SCHEDULED]",
      _core_cycle, _id);
  }
  for (const auto& inst : op->get_instructions()) {
    if (inst->is_ready())
      op->enqueue_ready(inst);
  }
  _tiles.push_back(std::move(op));
}


// Pop one finished tile if exist
std::shared_ptr<Tile> Core::pop_finished_tile() {
  std::shared_ptr<Tile> result = std::make_unique<Tile>(Tile(Tile::Status::EMPTY));
  if (_finished_tiles.size() > 0) {
    result = std::move(_finished_tiles.front());
    _finished_tiles.pop();
  }
  return result;
}


// Return true if the core still has tiles or pipeline/DMA work in flight
bool Core::running() {
  bool running = false;
  running = running || _tiles.size() > 0;
  running = running || !_vu_compute_pipeline.empty();
  for (int i=0; i<_num_systolic_array_per_core;i++)
    running = running || !_sa_compute_pipeline.at(i).empty();
  running = running || !_dma_waiting_queue.empty() || !_dma_finished_queue.empty();
  running = running || !_dma.empty();
  running = running || !_ld_inst_queue.empty();
  running = running || !_st_inst_queue.empty();
  return running;
}



// <Related to per-cycle execution>
// Process the vector‑unit pipeline, finishing instructions whose finish_cycle ≤ _core_cycle, updating VU stats
void Core::vu_cycle() {
  bool retry = true;
  while (retry) {
    if (!_vu_compute_pipeline.empty()) {
      _stat_vu_compute_cycle++;
      if(_vu_compute_pipeline.front()->finish_cycle <= _core_cycle) {
        int bubble = _vu_compute_pipeline.front()->bubble_cycle;
        _stat_vu_compute_idle_cycle += bubble;
        _stat_vu_compute_cycle -= bubble;
        finish_instruction(_vu_compute_pipeline.front());
        _vu_compute_pipeline.pop();
      } else {
        retry = false;
      }
    } else {
      _stat_vu_compute_idle_cycle++;
      retry = false;
    }
  }
}

// For each systolic array pipeline, similarly proceed one cycle and update SA stats
void Core::sa_cycle() {
  for (int i=0; i<_num_systolic_array_per_core; i++) {
    bool retry = true;
    while (retry) {
      if (!_sa_compute_pipeline.at(i).empty()) {
        if(_sa_compute_pipeline.at(i).front()->finish_cycle <= _core_cycle) {
          int bubble = _sa_compute_pipeline.at(i).front()->bubble_cycle;
          _stat_sa_compute_idle_cycle.at(i) += bubble;
          _stat_sa_compute_cycle.at(i) -= bubble;
          finish_instruction(_sa_compute_pipeline.at(i).front());
          _sa_compute_pipeline.at(i).pop();
        } else {
          _stat_sa_compute_cycle.at(i)++;
          retry = false;
        }
      } else {
        _stat_sa_compute_idle_cycle.at(i)++;
        retry = false;
      }
    }
  }
}


// Run vu_cycle() + sa_cycle() to advance vector and systolic pipelines
void Core::compute_cycle() {
  vu_cycle();
  sa_cycle();
}


// Complete finished DMA operations, update tag tables/barriers, issue the next DMA instruction if available, push new mem_fetches to _request_queue, and update DMA stats
void Core::dma_cycle() {
  /* Check finished dma operation */
  while(_dma_finished_queue.size()) {
    std::shared_ptr<Instruction>& instruction = _dma_finished_queue.at(0);
    assert(instruction->get_waiting_request()==0);

    /* Finish DMA read instruction */
    if (instruction->is_dma_read() && !instruction->is_async_dma())
      finish_instruction(instruction);

    /* Set tag table of async dma load */
    if (instruction->is_dma_read() && instruction->is_async_dma()) {
      auto& key = instruction->get_tag_id();
      assert(!_dma.get_tag_finish(instruction->subgraph_id, key));
      _dma.set_tag_finish(instruction->subgraph_id, key);
      spdlog::trace("[{}][Core {}] {} ASYNC FINISHED, subgraph_id: {} addr_name: {} tag_id: {} tag_idx_list: {} tag_stride_list: {}",
                    _core_cycle, _id, opcode_to_string(instruction->get_opcode()),
                    instruction->subgraph_id, instruction->get_addr_name(),
                    fmt::format("[{}]", fmt::join(instruction->get_tag_id(), ", ")),
                    fmt::format("[{}]", fmt::join(instruction->get_tag_idx_list(), ", ")),
                    fmt::format("[{}]", fmt::join(instruction->get_tag_stride_list(), ", ")));
      for (auto & wait_inst : _dma.get_tag_waiter(instruction->subgraph_id, key)) {
        _dma.mark_tag_used(instruction->subgraph_id, key);
        finish_instruction(wait_inst);
      }
    }
    _dma_finished_queue.erase(_dma_finished_queue.begin());
  }

  if (_dma.is_finished()) {
    /* Finish instruction when it is DMA store */
    if (_dma.get_current_inst() != nullptr) {
      std::shared_ptr<Instruction> finished_inst = std::move(_dma.get_current_inst());
      if (finished_inst->is_dma_write()) {
        /* Only DMA write operation is finished! */
        finish_instruction(finished_inst);
      } else if (finished_inst->is_dma_read() && finished_inst->is_async_dma()) {
        /* Register tag table for async dma load */
        _dma.register_tag(finished_inst->subgraph_id, finished_inst->get_tag_id());
        finish_instruction(finished_inst);
      } else if(!finished_inst->is_dma_read()) {
        spdlog::error("[{}][Core {}] DMA instruction in not valid", _core_cycle, _id);
        exit(EXIT_FAILURE);
      } else if (finished_inst->get_opcode() == Opcode::BAR) {
        spdlog::trace("[{}][Core {}] {} FINISHED, addr_name: {} tag_id: {} tag_idx_list: {} tag_stride_list: {}", _core_cycle, _id,
                      opcode_to_string(finished_inst->get_opcode()), finished_inst->get_addr_name(),
                      fmt::format("[{}]", fmt::join(finished_inst->get_tag_id(), ", ")),
                      fmt::format("[{}]", fmt::join(finished_inst->get_tag_idx_list(), ", ")),
                      fmt::format("[{}]", fmt::join(finished_inst->get_tag_stride_list(), ", ")));
      }
      /*Pass to waiting queue */
      _dma_waiting_queue[finished_inst.get()] = std::move(finished_inst);
    }

    /* Issue new DMA operation */
    if (!_ld_inst_queue.empty()) {
      std::shared_ptr<Instruction> inst = _ld_inst_queue.front();
      _dma.issue_tile(inst);
      _ld_inst_queue.pop();
    } else if (!_st_inst_queue.empty()) {
      std::shared_ptr<Instruction> inst = _st_inst_queue.front();
      _dma.issue_tile(inst);
      _st_inst_queue.pop();
    } else {
      /* DMA is idle */
      _stat_dma_idle_cycle++;
      return;
    }
  }
  /* Generate memfetch */
  auto access_vec = _dma.get_memory_access(_core_cycle, _config.icnt_injection_ports_per_core);
  for (auto access : *access_vec) {
    access->set_start_cycle(_core_cycle);
    _request_queue.push(access);
  }

  /* Increase dma stat cycle */
  _stat_dma_cycle++;
}


// Proceed one core cycle
// Iterate each active tiles of a core and the issue the instruction of the first tile that has ready instruction
// Run compute (compute_cycle), DMA (dma_cycle), increment _core_cycle, issue one instruction if possible, and move finished tiles to _finished_tiles
void Core::cycle() {
  /* Run compute unit and DMA unit */
  compute_cycle();
  dma_cycle();

  /* Increase core cycle counter */
  _core_cycle++;

  /* Iterate tile while an instruction is issued */
  bool issued = false;

  for (int i=0; i<_tiles.size() && !issued; i++) {
    auto& instructions = _tiles[i]->get_ready_instructions();
    for (auto it=instructions.begin(); it!=instructions.end();) {
      auto& inst = *it;
      /* Skip instruction is not ready  */
      //if (!inst->is_ready())
      //  continue;

      switch (inst->get_opcode()) {
        case Opcode::MOVIN:
          {
            /* Check another MOVIN with same tag is issued */
            auto& key = inst->get_tag_id();
            if (inst->is_sparse_inst()) {
              _dma.register_tag(inst->subgraph_id, key);
              _dma.set_tag_sparse(inst->subgraph_id, key);
              finish_instruction(inst);
              issued = true;
              _stat_tot_skipped_inst.at(static_cast<size_t>(inst->get_opcode()))++;
              break;
            } else if (inst->is_async_dma() && _dma.tag_key_exist(inst->subgraph_id, key)) {
              bool finished = _dma.get_tag_finish(inst->subgraph_id, key);
              if (finished)
                finish_instruction(inst);
              else
                _dma.register_tag_waiter(inst->subgraph_id, key, inst);
              spdlog::trace("[{}][Core {}][SIKIPPED] {}, addr_name: {} tag_id: {} tag_idx_list: {} tag_stride_list: {}", _core_cycle, _id,
                            opcode_to_string(inst->get_opcode()),
                            inst->get_addr_name(),
                            fmt::format("[{}]", fmt::join(inst->get_tag_id(), ", ")),
                            fmt::format("[{}]", fmt::join(inst->get_tag_idx_list(), ", ")),
                            fmt::format("[{}]", fmt::join(inst->get_tag_stride_list(), ", ")));
              issued = true;
              _stat_tot_skipped_inst.at(static_cast<size_t>(inst->get_opcode()))++;
              break;
            } else {
              spdlog::trace("[{}][Core {}][INST_ISSUED] {}, addr_name: {} tag_id: {} tag_idx_list: {} tag_stride_list: {}", _core_cycle, _id,
                            opcode_to_string(inst->get_opcode()),
                            inst->get_addr_name(),
                            fmt::format("[{}]", fmt::join(inst->get_tag_id(), ", ")),
                            fmt::format("[{}]", fmt::join(inst->get_tag_idx_list(), ", ")),
                            fmt::format("[{}]", fmt::join(inst->get_tag_stride_list(), ", ")));
              _ld_inst_queue.push(inst);
              issued = true;
              break;
            }
          }
        case Opcode::MOVOUT:
          spdlog::trace("[{}][Core {}][INST_ISSUED] {}, addr_name: {} tag_id: {} tag_idx_list: {} tag_stride_list: {}", _core_cycle, _id,
                        opcode_to_string(inst->get_opcode()),
                        inst->get_addr_name(),
                        fmt::format("[{}]", fmt::join(inst->get_tag_id(), ", ")),
                        fmt::format("[{}]", fmt::join(inst->get_tag_idx_list(), ", ")),
                        fmt::format("[{}]", fmt::join(inst->get_tag_stride_list(), ", ")));
          _st_inst_queue.push(inst);
          issued = true;
          break;
        case Opcode::COMP:
          {
            auto& target_pipeline = get_compute_pipeline(inst->get_compute_type());
            if (target_pipeline.empty()) {
              inst->finish_cycle = _core_cycle + inst->get_compute_cycle();
              inst->bubble_cycle = inst->get_overlapping_cycle();
            } else {
              int overlapped_cycle = std::min(target_pipeline.back()->finish_cycle - _core_cycle, inst->get_overlapping_cycle());
              int bubble_cycle = inst->get_overlapping_cycle() - overlapped_cycle;
              inst->finish_cycle = target_pipeline.back()->finish_cycle + inst->get_compute_cycle() - overlapped_cycle;
              inst->bubble_cycle = bubble_cycle;
            }

            if (inst->get_compute_cycle() == 0) {
              inst->finish_instruction();
              static_cast<Tile*>(inst->get_owner())->inc_finished_inst();
              _stat_tot_skipped_inst.at(static_cast<size_t>(inst->get_opcode()))++;
              instructions.erase(it);
            } else {
              spdlog::trace("[{}][Core {}][INST_ISSUED][SA {}] {}-{}, finsh at {}", _core_cycle, _id, _systolic_array_rr,
                            opcode_to_string(inst->get_opcode()), inst->get_compute_type(), inst->finish_cycle);
              target_pipeline.push(inst);
              issued = true;
              if (inst->get_compute_type()) {
                _stat_gemm_inst++;
              }
            }
          }
          break;
        case Opcode::BAR:
          {
            auto& key = inst->get_tag_id();
            uint32_t finished = _dma.get_tag_finish(inst->subgraph_id, key);
            if (finished == -1) {
              for (auto child_inst : inst->get_child_inst()) {
                if (child_inst->get_opcode() == Opcode::COMP && child_inst->get_compute_type() == MATMUL) {
                  child_inst->set_compute_cycle(0);
                }
              }
              finish_instruction(inst);
            } else if (finished != 0) {
              _dma.mark_tag_used(inst->subgraph_id, key);
              finish_instruction(inst);
            } else {
              _dma.register_tag_waiter(inst->subgraph_id, key, inst);
            }
            spdlog::trace("[{}][Core {}][INST_ISSUED] {},  addr_name: {} tag_id: {} tag_idx_list: {} tag_stride_list: {}", _core_cycle, _id,
                            opcode_to_string(inst->get_opcode()), inst->get_addr_name(),
                            fmt::format("[{}]", fmt::join(inst->get_tag_id(), ", ")),
                            fmt::format("[{}]", fmt::join(inst->get_tag_idx_list(), ", ")),
                            fmt::format("[{}]", fmt::join(inst->get_tag_stride_list(), ", ")));
            issued = true;
          }
          break;
        default:
          spdlog::error("Undefined instruction opcode type");
          exit(EXIT_FAILURE);
      }

      if (issued) {
        _stat_inst_count.at(static_cast<size_t>(inst->get_opcode()))++;
        instructions.erase(it);
        break;
      }
      it++;
    }
  }

  /* Remove finshed tiles */
  bool retry = true;
  while (retry) {
    for (int i=0; i<_tiles.size() && !issued; i++) {
      if (_tiles[i]->all_insts_finshed()) {
        _tiles[i]->set_status(Tile::Status::FINISH);
        _finished_tiles.push(std::move(_tiles[i]));
        _tiles.erase(_tiles.begin() + i); // FIXME. Inefficient data structure
        /* Let's retry */
        break;
      }
    }
    retry = false;
  }
  if(_config.core_print_interval && _core_cycle % _config.core_print_interval == 0) {
    print_current_stats();
  }
}

// <Related to Instruction Handling>
// Mark an instruction as finished, increment its owning tile’s finished‑inst counter, and log it (with special handling for async DMA / COMP)
void Core::finish_instruction(std::shared_ptr<Instruction>& inst) {
  if (inst->finished) {
    spdlog::error("[{}][Core {}][ERROR] {} inst already finished!!", _core_cycle, _id,
                  opcode_to_string(inst->get_opcode()));
    exit(EXIT_FAILURE);
  }
  inst->finish_instruction();
  static_cast<Tile*>(inst->get_owner())->inc_finished_inst();
  if (inst->get_opcode() == Opcode::COMP) {
    spdlog::trace("[{}][Core {}][INST_FINISHED] {}-{}",
      _core_cycle, _id, opcode_to_string(inst->get_opcode()), inst->get_compute_type());
  } else if (inst->get_opcode() != Opcode::BAR && inst->is_async_dma()){
    spdlog::trace("[{}][Core {}][ASYNC] {} subgraph_id: {} addr_name: {} tag_id: {} tag_idx_list: {} tag_stride_list: {}",
      _core_cycle, _id, opcode_to_string(inst->get_opcode()), inst->subgraph_id, inst->get_addr_name(),
      inst->get_tag_id(),
      fmt::format("[{}]", fmt::join(inst->get_tag_idx_list(), ", ")),
      fmt::format("[{}]", fmt::join(inst->get_tag_stride_list(), ", ")));
  } else if ((inst->get_opcode() == Opcode::MOVIN || inst->get_opcode() == Opcode::MOVOUT) && !inst->is_async_dma()) {
    spdlog::trace("[{}][Core {}][INST_FINISHED] {} addr_name: {}", _core_cycle, _id,
      opcode_to_string(inst->get_opcode()), inst->get_addr_name());
  }
}

// Returns the compute pipeline queue for the given compute_type.
// VECTOR_UNIT → shared VU queue. MATMUL/PRELOAD → next SA queue (round-robin across SAs).
// Each pipeline is a timing-abstraction queue: instructions sit in it with a pre-stamped
// finish_cycle and are popped by sa_cycle()/vu_cycle() when that cycle is reached.
std::queue<std::shared_ptr<Instruction>>& Core::get_compute_pipeline(int compute_type) {
  if (compute_type == VECTOR_UNIT)
    return _vu_compute_pipeline;
  else if (compute_type == MATMUL || compute_type == PRELOAD) {
    uint32_t sa_idx = _systolic_array_rr;
    _systolic_array_rr = (_systolic_array_rr + 1) % _num_systolic_array_per_core;
    return _sa_compute_pipeline.at(sa_idx);
  }
  else {
    spdlog::error("Undefined compute type");
    exit(EXIT_FAILURE);
  }
}


// Return whether a compute instruction is ready to be issued (essentially inst->is_ready())
bool Core::can_issue_compute(std::shared_ptr<Instruction>& inst) {
  return inst->is_ready();
}





//<Related to memory request/response (to/from interconnect)>
// Return whether there are mem_fetches in _request_queue
bool Core::has_memory_request() {
  return !_request_queue.empty();
}


// Pop the front request after the interconnect has consumed it
void Core::pop_memory_request() {
  _request_queue.pop();
}


// Consume a response, decrement the owning instruction’s outstanding request counter, and once all responses are in, move the instruction into _dma_finished_queue
void Core::push_memory_response(mem_fetch* response) {
  Instruction* owner_inst = static_cast<Instruction*>(response->get_custom_data());
  assert(owner_inst->get_waiting_request());

  owner_inst->dec_waiting_request();
  if (!owner_inst->get_waiting_request()) {
    auto it = _dma_waiting_queue.find(owner_inst);
    if (it != _dma_waiting_queue.end()) {
      std::shared_ptr<Instruction> moved_inst = std::move(it->second);
      _dma_finished_queue.push_back(std::move(moved_inst));
      _dma_waiting_queue.erase(it);
    } else {
      assert(true || "Can't happend...!");
    }
  }
  _stat_mem_response++;
  delete response;
}


// <Related to statistics reporting>
// Print total instruction counts, skipped counts, GEMM vs vector breakdown, per‑SA utilization, DMA bandwidth/usage, VU utilization, NUMA stats, and total cycles
void Core::print_stats() {
  std::vector<float> sa_utilization;
  update_stats();
  spdlog::info("===== Instructions count =====");
  for (int i = 0; i < static_cast<size_t>(Opcode::COUNT); i++) {
    auto opcode  = static_cast<Opcode>(i);
    auto inst = _stat_inst_count.at(i);
    auto skipped = _stat_tot_skipped_inst.at(i);
    auto name = opcode_to_string(opcode);

    if (opcode == Opcode::COMP) {
      auto gemm   = _stat_gemm_inst;
      auto vector = inst - gemm;
      if (skipped)
        spdlog::info("Core [{}] : {:8} inst_count {} (GEMM: {}, Vector: {}), skipped inst_count {}",
            _id, name, inst, gemm, vector, skipped);
      else
        spdlog::info("Core [{}] : {:8} inst_count {} (GEMM: {}, Vector: {})",
            _id, name, inst, gemm, vector);
    }
    else {
      if (skipped)
        spdlog::info("Core [{}] : {:8} inst_count {}, skipped inst_count {}",
            _id, name, inst, skipped);
      else
        spdlog::info("Core [{}] : {:8} inst_count {}",
            _id, name, inst);
    }
  }
  spdlog::info("========= Core stat =========");
  for (int i=0; i<_num_systolic_array_per_core; i++)
    sa_utilization.push_back(static_cast<float>(_stat_tot_sa_compute_cycle.at(i) * 100) / _core_cycle);
  for (int i=0; i<_num_systolic_array_per_core; i++)
    spdlog::info("Core [{}] : Systolic array [{}] utilization(%) {:.2f}, active_cycles {}, idle_cycles {}", _id, i, sa_utilization.at(i),
      _stat_tot_sa_compute_cycle.at(i), _stat_tot_sa_compute_idle_cycle.at(i));
  float dram_bw = _config.dram_req_size * _stat_tot_mem_response * _config.core_freq_mhz / (_core_cycle * 1000); // B/cycle
  spdlog::info("Core [{}] : DMA active_cycles, {} DMA idle_cycles {}, DRAM BW {:.3f} GB/s ({} responses)", _id, _stat_tot_dma_cycle, _stat_tot_dma_idle_cycle, dram_bw, _stat_tot_mem_response);
  spdlog::info("Core [{}] : Vector unit utilization(%) {:.2f}, active cycle {}, idle_cycle {}", _id,
    static_cast<float>(_stat_tot_vu_compute_cycle * 100) / _core_cycle, _stat_tot_vu_compute_cycle, _stat_tot_vu_compute_idle_cycle);
  spdlog::info("Core [{}] : NUMA local memory: {} requests, remote memory: {} requests", _id, _stat_numa_local_access, _stat_numa_remote_access);
  spdlog::info("Core [{}] : Total_cycles {}", _id, _core_cycle);
}


// Prints stats over the last core_print_interval cycles and then call update_stats()
// print_stats() but only for the last few intervals
void Core::print_current_stats() {
  std::vector<float> sa_utilization;
  for (int i=0; i<_num_systolic_array_per_core; i++)
    sa_utilization.push_back(static_cast<float>(_stat_sa_compute_cycle.at(i) * 100) / _config.core_print_interval);
  float dram_bw = _config.dram_req_size * _stat_mem_response * _config.core_freq_mhz / (_config.core_print_interval * 1000); // B/cycle
  auto level = spdlog::level::info;
  if(_id != 0)
    level = spdlog::level::debug;

  spdlog::info("========= Core stat =========");
  for (int i=0; i<_num_systolic_array_per_core; i++)
    spdlog::info("Core [{}] : Systolic array [{}] utilization(%) {:.2f}, active_cycles {}, idle_cycles {}", _id, i, sa_utilization.at(i),
      _stat_sa_compute_cycle.at(i), _stat_sa_compute_idle_cycle.at(i));
  spdlog::info("Core [{}] : DMA active_cycles {}, DMA idle_cycles {}, DRAM BW {:.3f} GB/s ({} responses)", _id, _stat_dma_cycle, _stat_dma_idle_cycle, dram_bw, _stat_mem_response);
  spdlog::info("Core [{}] : Vector unit Utilization(%) {:.2f}, active_cycles {}, idle_cycles {}", _id,
    static_cast<float>(_stat_vu_compute_cycle * 100) / _config.core_print_interval, _stat_vu_compute_cycle, _stat_vu_compute_idle_cycle);
  spdlog::info("Core [{}] : Total_cycles {}", _id, _core_cycle);
  update_stats();
}


// Fold per‑interval stats into total stats and clear the interval counters
void Core::update_stats() {
  for (int i=0; i<_num_systolic_array_per_core; i++) {
    _stat_tot_sa_compute_cycle.at(i) += _stat_sa_compute_cycle.at(i);
    _stat_tot_sa_compute_idle_cycle.at(i) += _stat_sa_compute_idle_cycle.at(i);
    _stat_sa_compute_cycle.at(i) = 0;
    _stat_sa_compute_idle_cycle.at(i) = 0;
  }

  _stat_tot_vu_compute_cycle += _stat_vu_compute_cycle;
  _stat_tot_dma_cycle += _stat_dma_cycle;
  _stat_tot_dma_idle_cycle += _stat_dma_idle_cycle;
  _stat_tot_mem_response += +_stat_mem_response;

  _stat_vu_compute_cycle = 0;
  _stat_dma_cycle = 0;
  _stat_dma_idle_cycle = 0;
  _stat_vu_compute_idle_cycle = 0;
  _stat_mem_response = 0;
}