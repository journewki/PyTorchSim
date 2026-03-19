// Define Tile class, which is the target of operation in TOGSim. 
// Tile is like a set of instructions for one kernel. 
// Kernel in Python code -> torch.compile() -> preregisterd device = customized npu simulator generate TOG -> TOGSim get TOG and run simulation with it. 

#include "Tile.h"
#include "TileGraph.h"


// 
Tile::Tile(Status status) {
  _status = status;
}


// Increment the ready counter (tracks number of parent dependencies not yet completed)
void Tile::inc_ready_counter() {
  _ready_counter++;
}


// Decrement the ready counter when a parent tile finishes, with error checking for underflow
void Tile::dec_ready_counter() {
  if (_ready_counter==0) {
    spdlog::error("Tile ready counter is already 0...");
    exit(EXIT_FAILURE);
  }
  _ready_counter--;
}


// Add an instruction to this tile, sets the tile as owner, and increments instruction count
void Tile::append_instuction(std::shared_ptr<Instruction>& inst) {
  /* Move instructions */
  _nr_insts++;
  inst->set_owner(this);
  inst->set_owner_ready_queue(&_ready_queue);
  _instructions.push_back(inst);
}


// Register a child tile dependency and increment the child's ready counter
void Tile::append_child(std::shared_ptr<Tile> child) {
  child->inc_ready_counter();
  _child_tiles.push_back(std::move(child));
}

// Called when tile execution completes, decrements ready counters of all child tiles to signal they can proceed
void Tile::finish_tile() {
  for (auto& child_tile_ptr: _child_tiles)
    child_tile_ptr->dec_ready_counter();
}


// Log all instructions in this tile for debugging purposes
void Tile::print() {
  spdlog::info("Tile: [");
  for (const auto& inst: _instructions) {
    inst->print();
  }
  spdlog::info("]");
}