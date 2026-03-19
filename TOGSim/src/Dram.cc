// Define Dram class and DramRamulator2 and SimpleDRAM classes inheriting Dram class.
// DramRamulator2 uses Ramulator2 for cycle-accurate timing, while SimpleDRAM uses fixed-latency delay queues.
// Both manage per-channel L2 caches and queues for interconnect communication.
//
// Request flow through both implementations:
//   Interconnect → m_from_crossbar_queue → L2 cache (_m_caches) → DRAM (_mem)
//   Interconnect ← m_to_crossbar_queue  ← L2 cache (_m_caches) ← DRAM (_mem)
//
/*
=== Dram (base class) Attributes ===
`_config`
- Full SimulationConfig snapshot used throughout DRAM and cache initialization
`_m_cache_config`
- Parsed L2 cache geometry (sets, associativity, line size, sector size),
  initialized from the config string only when l2d_type == DATACACHE
`_n_ch`
- Total number of DRAM channels across all partitions
`_n_bl`
- DRAM burst length; controls how many bits are transferred per request
`_n_partitions`
- Number of NUMA partitions; each partition owns a contiguous slice of channels
`_n_ch_per_partition`
- Number of DRAM channels assigned to each NUMA partition
`_req_size`
- Size in bytes of one DRAM request (e.g. 32B); used for address alignment and BW calculation
`_cycles`
- Internal DRAM-side cycle counter (incremented independently of core cycles)
`_core_cycles`
- Pointer to the Simulator's global core cycle counter; used by L2 cache for latency tracking
`m_from_crossbar_queue`(crossbar == interconnect)
- Per-channel incoming request queues; holds mem_fetch* arriving from the interconnect,
  waiting to be forwarded to the L2 cache
`m_to_crossbar_queue`
- Per-channel outgoing response queues; holds completed mem_fetch* ready to be
  picked up by the interconnect and returned to the requesting core
`m_cache_latency_queue`
- Per-channel delay queues that model the fixed access latency of the L2 cache
  before a request reaches the actual DRAM backend
`m_to_mem_queue`
- Per-channel queues for requests that have passed through the L2 cache and
  are waiting to be issued to the underlying DRAM model
`_m_caches`
- Per-channel L2 cache instances; either NoL2Cache (bypass) or L2DataCache (set-associative).
  Sits between m_from_crossbar_queue and the DRAM backend, absorbing hits before
  they reach DRAM
*/

#include "Dram.h"
// Map a mem_fetch address to a physical DRAM channel ID
// Use ipoly hash on (addr / req_size) for good address distribution across channels
// For small channel counts (<16), hashes to 16 buckets then mods down to avoid bias
// Add a NUMA partition offset so each partition's requests stay within its own channel slice
uint32_t Dram::get_channel_id(mem_fetch* access) {
  uint32_t channel_id;
  if (_n_ch_per_partition >= 16)
    channel_id = ipoly_hash_function((new_addr_type)access->get_addr()/_req_size, 0, _n_ch_per_partition);
  else
    channel_id = ipoly_hash_function((new_addr_type)access->get_addr()/_req_size, 0, 16) % _n_ch_per_partition;

  channel_id += ((access->get_numa_id() % _n_partitions)* _n_ch_per_partition);
  return channel_id;
}

// Base constructor: Initialize per-channel crossbar queues and per-channel L2 cache instances
// If l2d_type == NOCACHE, install a pass-through NoL2Cache per channel
// If l2d_type == DATACACHE, parse the cache geometry string and installs an L2DataCache per channel
Dram::Dram(SimulationConfig config, cycle_type* core_cycle) {
  _core_cycles = core_cycle;
  _n_ch = config.dram_channels;
  _n_bl = config.dram_nbl;
  _req_size = config.dram_req_size;
  _n_partitions = config.dram_num_partitions;
  _n_ch_per_partition = config.dram_channels_per_partitions;
  _config = config;

  spdlog::info("[Config/DRAM] DRAM Bandwidth {} GB/s, Freq: {} MHz, Channels: {}, Request_size: {}B", config.max_dram_bandwidth(), config.dram_freq_mhz, _n_ch, _req_size);
  /* Initialize DRAM Channels */
  for (int ch = 0; ch < _n_ch; ch++) {
    m_to_crossbar_queue.push_back(std::queue<mem_fetch*>());
    m_from_crossbar_queue.push_back(std::queue<mem_fetch*>());
  }

  /* Initialize L2 cache */
  _m_caches.resize(_n_ch);
  if (config.l2d_type == L2CacheType::NOCACHE) {
    std::string name = "No cache";
    spdlog::info("[Config/L2Cache] No L2 cache");
    for (int ch = 0; ch < _n_ch; ch++)
      _m_caches[ch] = new NoL2Cache(name, _m_cache_config, ch, _core_cycles, &m_to_crossbar_queue[ch], &m_from_crossbar_queue[ch]);
  } else if (config.l2d_type == L2CacheType::DATACACHE) {
    std::string name = "L2 cache";
    _m_cache_config.init(config.l2d_config_str);
    spdlog::info("[Config/L2Cache] Total Size: {} KB, Partition Size: {} KB, Set: {}, Assoc: {}, Line Size: {}B Sector Size: {}B",
            _m_cache_config.get_total_size_in_kb() * _n_ch, _m_cache_config.get_total_size_in_kb(),
            _m_cache_config.get_num_sets(), _m_cache_config.get_num_assoc(),
            _m_cache_config.get_line_size(), _m_cache_config.get_sector_size());
    for (int ch = 0; ch < _n_ch; ch++)
      _m_caches[ch] = new L2DataCache(name, _m_cache_config, ch, _core_cycles, _config.l2d_hit_latency, _config.num_cores, &m_to_crossbar_queue[ch], &m_from_crossbar_queue[ch]);
  } else {
    spdlog::error("[Config/L2D] Invalid L2 cache type...!");
    exit(EXIT_FAILURE);
  }
}


/*
DRAM class that is implemented through Ramulator2. It works as intermediate between actual Ramulator2 objet and TOGSim
=== DramRamulator2 Attributes ===
`_mem`
- Per-channel Ramulator2 instances that model cycle-accurate DRAM timing,
  including row open/close policy, row hits/misses/conflicts, and bandwidth utilization
`_tx_log2`
- log2(req_size); used to strip the intra-request byte offset bits from an address
  before passing it to Ramulator2 (Ramulator2 works in units of transactions)
`_tx_ch_log2`
- log2(channels_per_partition) + _tx_log2; used to strip both the channel-select bits
  and byte-offset bits when remapping addresses for Ramulator2
*/

// Initialize one Ramulator2 instance per channel using the DRAM config file path
// Precompute _tx_log2 and _tx_ch_log2 for fast address remapping in push()
DramRamulator2::DramRamulator2(SimulationConfig config, cycle_type* core_cycle) : Dram(config, core_cycle) {
  /* Initialize DRAM Channels */
  _mem.resize(_n_ch);
  for (int ch = 0; ch < _n_ch; ch++) {
    _mem[ch] = std::make_unique<Ramulator2>(
      ch, _n_ch, config.dram_config_path, "Ramulator2", _config.dram_print_interval, _n_bl);
  }
  _tx_log2 = log2(_req_size);
  _tx_ch_log2 = log2(_n_ch_per_partition) + _tx_log2;
}

// Return true if any channel still has requests in-flight inside Ramulator2 or response waiting inside the L2 cache to be forwarded to the crossbar
bool DramRamulator2::running() {
  for (int ch = 0; ch < _n_ch; ch++) {
    if (mem_fetch* req = _mem[ch]->return_queue_top())
      return true;
    if (mem_fetch* req = _m_caches[ch]->top())
      return true;
  }
  return false;
}

// Advance Ramulator2 one cycle per channel
// Feed the front of the L2 cache output into Ramulator2 (cache → DRAM direction), and deliver Ramulator2 completions back into the L2 cache (DRAM → cache direction)
void DramRamulator2::cycle() {
  for (int ch = 0; ch < _n_ch; ch++) {
    _mem[ch]->cycle();

    // From Cache to DRAM
    if (mem_fetch* req = _m_caches[ch]->top()) {
      _mem[ch]->push(req);
      _m_caches[ch]->pop();
    }

    // From DRAM to Cache
    if (mem_fetch* req = _mem[ch]->return_queue_top()) {
      if(_m_caches[ch]->push(req))
        _mem[ch]->return_queue_pop();
    }
  }
}

// Advance each channel's L2 cache one cycle, processing hits/misses and moving completed requests toward m_to_crossbar_queue
void DramRamulator2::cache_cycle()  {
  for (int ch = 0; ch < _n_ch; ch++) {
    _m_caches[ch]->cycle();
  }
}

// Always return false; the incoming crossbar queue has infinite capacity
bool DramRamulator2::is_full(uint32_t cid, mem_fetch* request) {
  return false; //m_from_crossbar_queue[cid].full(); Infinite length
}

// Strip channel-select bits from the address (so Ramulator2 sees a flat per-channel address space), then enqueue the request into m_from_crossbar_queue for that channel
void DramRamulator2::push(uint32_t cid, mem_fetch* request) {
  addr_type target_addr = (request->get_addr() >> _tx_ch_log2) << _tx_log2;
  request->set_addr(target_addr);
  m_from_crossbar_queue[cid].push(request);
}


// Return true if there are no completed responses waiting in m_to_crossbar_queue for this channel
bool DramRamulator2::is_empty(uint32_t cid) {
  return m_to_crossbar_queue[cid].empty();
}

// Return the front completed response from m_to_crossbar_queue without removing it
mem_fetch* DramRamulator2::top(uint32_t cid) {
  assert(!is_empty(cid));
  return m_to_crossbar_queue[cid].front();
}

// Remove the front completed response from m_to_crossbar_queue after it has been consumed by the interconnect
void DramRamulator2::pop(uint32_t cid) {
  assert(!is_empty(cid));
  m_to_crossbar_queue[cid].pop();
}

// Print Ramulator2's internal statistics (row hits/misses/conflicts, BW utilization) for each channel
void DramRamulator2::print_stat() {
  for (int ch = 0; ch < _n_ch; ch++) {
    _mem[ch]->print(stdout);
  }
}

// Print L2 cache hit/miss statistics for each channel.
void DramRamulator2::print_cache_stats() {
  for (int ch = 0; ch < _n_ch; ch++) {
    _m_caches[ch]->print_stats();
  }
}


/*
Simple version of DRAM object which uses fixed latency 
=== SimpleDRAM Attributes ===
`_latency`
- Fixed round-trip DRAM latency in cycles applied to every request uniformly,
  regardless of row state; trades accuracy for simulation speed
`_mem`
- Per-channel DelayQueue instances; each request pushed with _latency cycles
  and delivered to the L2 cache after exactly that many cycles have elapsed
`_tx_log2`
- log2(req_size); precomputed for address alignment (unused in push() for SimpleDRAM
  but kept symmetric with DramRamulator2)
`_tx_ch_log2`
- log2(channels_per_partition) + _tx_log2; precomputed for potential address remapping
*/

// Initialize one DelayQueue per channel with the configured fixed DRAM latency
// Precompute _tx_log2 and _tx_ch_log2 for address handling consistency with DramRamulator2
SimpleDRAM::SimpleDRAM(SimulationConfig config, cycle_type* core_cycle) : Dram(config, core_cycle) {
  /* Initialize DRAM Channels */
  spdlog::info("[SimpleDRAM] DRAM latecny: {}", config.dram_latency);
  for (int ch = 0; ch < _n_ch; ch++) {
    _mem.push_back(std::make_unique<DelayQueue<mem_fetch*>>("SimpleDRAM", true, -1));
  }
  _latency =  config.dram_latency;
  _tx_log2 = log2(_req_size);
  _tx_ch_log2 = log2(_n_ch_per_partition) + _tx_log2;
}

// Return true if any channel's DelayQueue still has in-flight requests or its L2 cache has responses pending for the crossbar
bool SimpleDRAM::running() {
  for (int ch = 0; ch < _n_ch; ch++) {
    if (!_mem[ch]->queue_empty())
      return true;
    if (mem_fetch* req = _m_caches[ch]->top())
      return true;
  }
  return false;
}

// Advance each channel's DelayQueue one cycle
// Feed the front of the L2 cache output into the DelayQueue with _latency (cache → DRAM), and delivers requests that have waited _latency cycles back into the L2 cache as replies (DRAM → cache)
void SimpleDRAM::cycle() {
  for (int ch = 0; ch < _n_ch; ch++) {
    _mem[ch]->cycle();

    // From Cache to DRAM
    if (mem_fetch* req = _m_caches[ch]->top()) {
      //spdlog::info("[Cache->DRAM] mem_fetch: addr={:#x}", req->get_addr());

      _mem[ch]->push(req, _latency);
      _m_caches[ch]->pop();
    }

    // From DRAM to Cache
    if (_mem[ch]->arrived()) {
      mem_fetch* req = _mem[ch]->top();
      req->set_reply();
      //spdlog::info("[DRAM->Cache] mem_fetch: addr={:#x}", req->get_addr());
      if(_m_caches[ch]->push(req))
        _mem[ch]->pop();
    }
  }
}

// Advance each channel's L2 cache one cycle, processing hits/misses and moving completed requests toward m_to_crossbar_queue
void SimpleDRAM::cache_cycle()  {
  for (int ch = 0; ch < _n_ch; ch++) {
    _m_caches[ch]->cycle();
  }
}

// Always return false; the incoming crossbar queue has infinite capacity
bool SimpleDRAM::is_full(uint32_t cid, mem_fetch* request) {
  return false; //m_from_crossbar_queue[cid].full(); Infinite length
}

// Enqueue the request directly into m_from_crossbar_queue for the given channel (no address remapping needed since SimpleDRAM does not use Ramulator2's address space)
void SimpleDRAM::push(uint32_t cid, mem_fetch* request) {
  m_from_crossbar_queue[cid].push(request);
}

// Return true if there are no completed responses waiting in m_to_crossbar_queue for this channel
bool SimpleDRAM::is_empty(uint32_t cid) {
  return m_to_crossbar_queue[cid].empty();
}


// Return the front completed response from m_to_crossbar_queue without removing it
mem_fetch* SimpleDRAM::top(uint32_t cid) {
  assert(!is_empty(cid));
  return m_to_crossbar_queue[cid].front();
}

// Remove the front completed response from m_to_crossbar_queue after it has been consumed by the interconnect
void SimpleDRAM::pop(uint32_t cid) {
  assert(!is_empty(cid));
  m_to_crossbar_queue[cid].pop();
}

// No-op: SimpleDRAM has no internal DRAM model with statistics to print.
void SimpleDRAM::print_stat() {}

// Prints L2 cache hit/miss statistics for each channel
void SimpleDRAM::print_cache_stats() {
  for (int ch = 0; ch < _n_ch; ch++) {
    _m_caches[ch]->print_stats();
  }
}
