// Define tile-level scheduling to each core. This is the extension for TOGSim from ONNXSim.
// Scheduler manages a FIFO queue of TileGraphs (one per kernel/op) and feeds individual
// Tiles to hardware cores cycle by cycle



#include "Scheduler.h"

// Store a pointer to the core's live cycle counter so scheduling
// Decisions can be made relative to the current simulation time
Scheduler::Scheduler(SimulationConfig config, const cycle_type* core_cycle, const uint64_t* core_time, int id)
    : _id(id), _config(config), _core_cycle(core_cycle), _core_time(core_time) {
}


// Add a new TileGraph (i.e. one compiled kernel/op) to the back of the queue, then checks whether the front graph has already finished and should be removed
void Scheduler::enqueue_graph(std::unique_ptr<TileGraph> tile_graph) {
  _tile_graph.push_back(std::move(tile_graph));
  refresh_status();
}

// Returns the next tile for (core_id, slot_id) without consuming it.
// Returns an EMPTY tile if: the queue is empty, the front graph hasn't arrived yet,
// or the graph's core type (WS_MESH vs STONNE) doesn't match the requesting core.
const std::shared_ptr<Tile> Scheduler::peek_tile(int core_id, int slot_id, CoreType ctype) {
  if (_tile_graph.empty() || _tile_graph.at(0)->get_arrival_time() > *_core_cycle)
    return std::make_unique<Tile>(Tile(Tile::Status::EMPTY));
  if ((!_tile_graph.at(0)->StonneGraph && ctype == CoreType::WS_MESH) || (_tile_graph.at(0)->StonneGraph && ctype == CoreType::STONNE))
    return _tile_graph.at(0)->peek_tile(core_id, slot_id);
  return std::make_unique<Tile>(Tile(Tile::Status::EMPTY));
}

// Consumes and returns the next tile for (core_id, slot_id).
// Records the start cycle the first time a real tile is issued from the front graph.
// Calls refresh_status() after each dispatch to pop finished graphs.
std::shared_ptr<Tile> Scheduler::get_tile(int core_id, int slot_id) {
  std::shared_ptr<Tile> tile = std::make_unique<Tile>(Tile(Tile::Status::EMPTY));
  if (empty(core_id)) {
    return tile;
  } else {
    tile = std::move(_tile_graph.at(0)->get_tile(core_id, slot_id));
     // Record start_time when first non-EMPTY tile is issued
    if (tile->get_status() != Tile::Status::EMPTY && _tile_graph.at(0)->get_start_time() == 0) {
      _tile_graph.at(0)->set_start_time(*_core_cycle);
    }
  }
  refresh_status();
  return tile;
}

// Returns true if the entire queue is empty (no pending graphs).
bool Scheduler::empty() {
  if (_tile_graph.empty())
    return true;
  return false;
}

// Returns true if there are no more tiles for the given core_id in the front graph,
// or if the queue is empty.
bool Scheduler::empty(int core_id) {
  if (_tile_graph.empty())
    return true;
  return _tile_graph.at(0)->empty(core_id);
}

// Checks whether the front TileGraph has finished executing.
// If so, logs the kernel's start cycle and total compute cycles, then removes it
// from the queue so the next graph can start being dispatched.
void Scheduler::refresh_status() {
  if (_tile_graph.empty())
    return;

  /* Remove finished request */
  if (_tile_graph.at(0)->is_finished()) {
    unsigned int kernel_id = _tile_graph.at(0)->get_kernel_id();
    cycle_type start_time = _tile_graph.at(0)->get_start_time();
    cycle_type compute_time = 0;
    if (start_time > 0) {
      compute_time = *_core_cycle - start_time;
    } else {
      // Fallback to arrival_time if start_time was not recorded
      start_time = _tile_graph.at(0)->get_arrival_time();
      compute_time = *_core_cycle - start_time;
    }

    spdlog::info("[Scheduler {}] Kernel {} has completed - TOG path: {} operation: {} finished at cycle {}",
                 _id, kernel_id, _tile_graph.at(0)->get_graph_path(),
                 _tile_graph.at(0)->get_name(), *_core_cycle);
    spdlog::info("[Scheduler {}] Kernel {} execution summary - Started at: {} cycles, Total compute time: {} cycles",
                 _id, kernel_id, start_time, compute_time);
    _tile_graph.pop_front();
  }
}