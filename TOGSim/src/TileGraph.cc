// Define TileSubGraph and TileGraph class, which is a graph of tiles, the unit for management of tiles with dependencies(subgraph) or within same core/slot pairs(graph)

#include "TileGraph.h"

// TileSubGraph is a subgraph of tiles with dependency management.
// One TileSubGraph is allocated to one (core,slot) 
// Using TileSubGraph, we can manage a group of related tiles with dependencies, tracking which tiles are ready to execute.


/*
Attributes
`_ready_tile_queue`
- Priority queue of tiles ready to execute (all dependencies met)
`_tile_set`
- Set of tiles still waiting for dependencies
`_id`
- Unique subgraph identifier
`_core_id`
- ID of the core this subgraph is assigned to (-1 if unassigned)
`_next_id`
- Static counter for generating unique subgraph IDs
`_cache_plan`
- IntervalTree for checking if memory regions are cacheable
*/


int TileSubGraph::_next_id = 0;
TileSubGraph::TileSubGraph() : _ready_tile_queue(), _tile_set(), _id(_next_id++) {
}


// Assign subgraph id for every instructions in a tile and add a tile to either ready queue (if no dependencies) or waiting set
void TileSubGraph::add_tile(std::shared_ptr<Tile> tile) {
  for (auto& inst : tile->get_instructions())
    inst->subgraph_id = _id;
  if (tile->get_ready_counter() == 0) {
   _ready_tile_queue.push(tile);
  } else {
    _tile_set.insert(tile);
  }
}


// Finish a tile and check if its children are now ready and moves them to ready queue
void TileSubGraph::finish_tile(std::shared_ptr<Tile> tile) {
  /* TODO. */
  tile->finish_tile();
  for (auto child_tile_ptr: tile->get_child_tile()) {
    if (child_tile_ptr->get_ready_counter())
      continue;
    /* if child is ready, add ready queue */
    _ready_tile_queue.push(child_tile_ptr);
    _tile_set.erase(child_tile_ptr);
  }
  return;
}


// Return the next ready tile without removing it from queue
const std::shared_ptr<Tile> TileSubGraph::peek_tile() {
  std::shared_ptr<Tile> ret = std::make_shared<Tile>(Tile::Status::EMPTY);
  if (_ready_tile_queue.empty())
    return ret;
  return _ready_tile_queue.top();
}


// Return the next ready tile with removing it from queue
std::shared_ptr<Tile> TileSubGraph::get_tile() {
  if (_ready_tile_queue.empty()) {
    std::shared_ptr<Tile> ret = std::make_shared<Tile>(Tile::Status::EMPTY);
    return ret;
  } else {
    std::shared_ptr<Tile> ret = _ready_tile_queue.top();
    _ready_tile_queue.pop();
    return ret;
  }
}





// TileGraph
// TileGraph is a collection of subgraphs mapped to cores/slots. We can manage multiple TileSubGraphs and assigns them to specific core/slot pairs for execution.
/*
Attributes
`_vec_index`
- Current index in subgraph vector (for tracking allocation progress)
`_path`
- File path to the graph definition
`_name`: 
- Name of this graph
`_loop_index_list`
- List of loop variable names for iteration
`_ranges`
- Vector of (start, end, step) tuples defining loop iteration ranges
`_subgraph_vec`
- Pool of unallocated subgraphs
`_finished_subgraph_vec`
- Completed subgraphs
`_cpu_graph_map[core_id][slot_id]`
- Maps of core/slot pairs to their currently assigned subgraph
`_cache_plan`
- IntervalTree for cache planning across all subgraphs
`_arrival_time`
- Cycle at which this graph becomes available for execution
`StonneGraph`
- Boolean flag indicating if this uses STONNE sparse core architecture
`null_tile`
- Static shared empty tile
*/


// Add a subgraph to _subgraph_vec and initializes its cache plan
void TileGraph::append_subgraph(std::shared_ptr<TileSubGraph> subgraph) {
  subgraph->init_cache_plan(_cache_plan);
  _subgraph_vec.push_back(std::move(subgraph));
}


// Check if all subgraphs are allocated and completed
bool TileGraph::is_finished() {
  bool finished = _subgraph_vec.empty();
  /* Check all outer loop is allocated */
  if (!finished)
    return finished;

  /* Check allocated subgraph is finished */
  for (const auto& core_pair: _cpu_graph_map) {
    for (const auto& tile_pair: core_pair.second)
      if (tile_pair.second != nullptr)
        finished &= tile_pair.second->is_finished();
  }
  return finished;
}


// Get next tile for a core/slot, allocating new subgraph if needed by calling allocate_subgraph()
const std::shared_ptr<Tile> TileGraph::peek_tile(int core_id, int slot_id) {
  std::shared_ptr<Tile> ret = std::make_unique<Tile>(Tile(Tile::Status::EMPTY));
  if (_cpu_graph_map.find(core_id) == _cpu_graph_map.end()) {
    allocate_subgraph(core_id, slot_id);
    return ret;
  } else if (_cpu_graph_map[core_id].find(slot_id) == _cpu_graph_map[core_id].end()) {
    allocate_subgraph(core_id, slot_id);
    return ret;
  } else if (_cpu_graph_map[core_id][slot_id] == nullptr) {
    allocate_subgraph(core_id, slot_id);
    return ret;
  }

  if (_cpu_graph_map[core_id][slot_id]->is_finished()){
    allocate_subgraph(core_id, slot_id);
    return ret;
  }
  return _cpu_graph_map[core_id][slot_id]->peek_tile();
}


// Return the next ready tile for the given (core, slot). 
// If the assigned subgraph is missing or finished, allocates the next one from the pool and returns EMPTY tile will be available next cycle). 
// Otherwise pop and return the highest-priority ready tilefrom the subgraph's queue
std::shared_ptr<Tile> TileGraph::get_tile(int core_id, int slot_id) {
  std::shared_ptr<Tile> ret = std::make_unique<Tile>(Tile(Tile::Status::EMPTY));
  if (_cpu_graph_map.find(core_id) == _cpu_graph_map.end()) {
    allocate_subgraph(core_id, slot_id);
    return ret;
  } else if (_cpu_graph_map[core_id].find(slot_id) == _cpu_graph_map[core_id].end()) {
    allocate_subgraph(core_id, slot_id);
    return ret;
  }

  if (_cpu_graph_map[core_id][slot_id]->is_finished()) {
    allocate_subgraph(core_id, slot_id);
    return ret;
  }
  return _cpu_graph_map[core_id][slot_id]->get_tile();
}

// Assign an available subgraph from _subgraph_vec to _cpu_graph_map[core_id][slot_id] 
void TileGraph::allocate_subgraph(int core_id, int slot_id) {
  if (_cpu_graph_map[core_id][slot_id] != nullptr) {
    _finished_subgraph_vec.push_back(_cpu_graph_map[core_id][slot_id]);
    _cpu_graph_map[core_id][slot_id] = nullptr;
  }

  for (auto it = _subgraph_vec.begin(); it != _subgraph_vec.end(); ++it) {
    if ((*it)->get_core_id() == -1 || (*it)->get_core_id() == core_id) {
      std::shared_ptr<TileSubGraph> subgraph = *it;
      _cpu_graph_map[core_id][slot_id] = subgraph;
      _subgraph_vec.erase(it);
      return;
    }
  }
  return;
}