// Define CacheStats class which collects and reports cache performance metrics
// Track hit/miss counts broken down by memory access type (read/write/writeback/etc.) and cache request outcome (HIT, MISS, SECTOR_MISS, HIT_RESERVED, RESERVATION_FAIL, MSHR_HIT)
// Also track separate failure reason counts for reservation failures
//
/*
=== CacheStats Attributes ===
`m_stats`
- 2D table [NUM_MEM_ACCESS_TYPE][NUM_CACHE_REQUEST_STATUS] counting every cache access outcome.
  Rows are access types (GLOBAL_ACC_R, GLOBAL_ACC_W, L2_CACHE_WA, L2_CACHE_WB, etc.),
  columns are outcomes (HIT, MISS, SECTOR_MISS, HIT_RESERVED, RESERVATION_FAIL, MSHR_HIT)
`m_fail_stats`
- 2D table [NUM_MEM_ACCESS_TYPE][NUM_CACHE_RESERVATION_FAIL_REASON] counting reservation
  failure reasons separately (e.g. MSHR full, miss queue full, data port busy)
`m_cache_port_available_cycles`
- Total cycles in which the cache data port was available (not occupied by a request);
  used for port utilization calculation
`m_cache_data_port_busy_cycles`
- Total cycles in which the cache data port was occupied serving a read/write access;
  complements m_cache_port_available_cycles to compute utilization ratio
`m_cache_fill_port_busy_cycles`
- Total cycles in which the cache fill port was busy receiving a line fill from DRAM;
  tracks fill bandwidth consumption separately from regular data port usage
`m_prev_hit`
- Snapshot of get_hit() taken at the last get_interval_hit() call; used to compute per-interval hit delta without storing a separate counter
`m_prev_miss`
- Snapshot of get_miss() taken at the last get_interval_miss() call; used to compute per-interval miss delta without storing a separate counter
*/

#include "Cache_stats.h"
#include "Memfetch.h"


// Initialize all stat tables to zero and resets port cycle counters and interval snapshots
// m_stats is sized [NUM_MEM_ACCESS_TYPE][NUM_CACHE_REQUEST_STATUS], m_fail_stats is sized [NUM_MEM_ACCESS_TYPE][NUM_CACHE_RESERVATION_FAIL_REASON]
CacheStats::CacheStats() {
  m_stats.resize(NUM_MEM_ACCESS_TYPE);
  m_fail_stats.resize(NUM_MEM_ACCESS_TYPE);
  for (int i = 0; i < NUM_MEM_ACCESS_TYPE; i++) {
    m_stats[i].resize(NUM_CACHE_REQUEST_STATUS, 0);
    m_fail_stats[i].resize(NUM_CACHE_RESERVATION_FAIL_REASON, 0);
  }
  m_cache_port_available_cycles = 0;
  m_cache_data_port_busy_cycles = 0;
  m_cache_fill_port_busy_cycles = 0;

  m_prev_hit = 0;
  m_prev_miss = 0;
}

// Reset all hit/miss counts and port cycle counters to zero without reallocating the tables
// Used to reset stats between measurement windows
void CacheStats::clear() {
  for (int i = 0; i < NUM_MEM_ACCESS_TYPE; i++) {
    std::fill(m_stats[i].begin(), m_stats[i].end(), 0);
    std::fill(m_fail_stats[i].begin(), m_fail_stats[i].end(), 0);
  }
  m_cache_port_available_cycles = 0;
  m_cache_data_port_busy_cycles = 0;
  m_cache_fill_port_busy_cycles = 0;
}


// Increment m_stats[access_type][access_outcome] by one
// Called on every cache access (hit or miss) to record its outcome
void CacheStats::inc_stats(int access_type, int access_outcome) {
  assert(check_valid(access_type, access_outcome));
  m_stats[access_type][access_outcome]++;
}

// Increment m_fail_stats[access_type][fail_outcome] by one
// Called specifically when a cache access results in RESERVATION_FAIL, recording the detailed reason (e.g. MSHR full, miss queue full)
void CacheStats::inc_fail_stats(int access_type, int fail_outcome) {
  assert(check_fail_valid(access_type, fail_outcome));
  m_fail_stats[access_type][fail_outcome]++;
}

// Resolve which status to record when a probe status and an access status differ
// 1) Probe — a non-destructive tag lookup to check if the line exists (probe_mode=true). Returns a status like HIT_RESERVED or SECTOR_MISS. 
// 2) Access — the actual tag lookup that allocates MSHR, sends miss requests, etc. Returns its own status.
// The probe can give more precise information (e.g. it was a sector miss, not a full miss), so in some cases the probe status is more accurate to record than the access status. This function picks whichever is more informativ
// HIT_RESERVED from probe takes precedence unless the access resulted in RESERVATION_FAIL
// SECTOR_MISS from probe takes precedence if the access confirmed a full MISS
// Otherwise, the access status is used directly
CacheRequestStatus CacheStats::select_stats_status(
    CacheRequestStatus probe, CacheRequestStatus access) const {
  if (probe == HIT_RESERVED && access != RESERVATION_FAIL)
    return probe;
  else if (probe == SECTOR_MISS && access == MISS)
    return probe;
  else
    return access;
}

// Return a mutable reference to m_stats or m_fail_stats at [access_type][access_outcome]
// If fail_outcome is true, indexes into m_fail_stats (reservation failure reasons);
// otherwise indexes into m_stats (normal hit/miss outcomes)
uint64_t &CacheStats::operator()(int access_type, int access_outcome,
                                 bool fail_outcome) {
  if (fail_outcome) {
    assert(check_fail_valid(access_type, access_outcome));
    return m_fail_stats[access_type][access_outcome];
  } else {
    assert(check_valid(access_type, access_outcome));
    return m_stats[access_type][access_outcome];
  }
}


// Const version of operator(): returns the value (not a reference) from m_stats or m_fail_stats
uint64_t CacheStats::operator()(int access_type, int access_outcome,
                                bool fail_outcome) const {
  if (fail_outcome) {
    assert(check_fail_valid(access_type, access_outcome));
    return m_fail_stats[access_type][access_outcome];
  } else {
    assert(check_valid(access_type, access_outcome));
    return m_stats[access_type][access_outcome];
  }
}

// Return a new CacheStats that is the element-wise sum of this and other
// Combine both m_stats and m_fail_stats tables and all port cycle counters
// Used to aggregate stats across multiple cache instances (e.g. per-channel caches)
CacheStats CacheStats::operator+(const CacheStats &other) {
  CacheStats sum;
  for (int i = 0; i < NUM_MEM_ACCESS_TYPE; i++) {
    for (int j = 0; j < NUM_CACHE_REQUEST_STATUS; j++) {
      sum.m_stats[i][j] = m_stats[i][j] + other.m_stats[i][j];
      sum.m_fail_stats[i][j] = m_fail_stats[i][j] + other.m_fail_stats[i][j];
    }
  }
  sum.m_cache_port_available_cycles =
      m_cache_port_available_cycles + other.m_cache_port_available_cycles;
  sum.m_cache_data_port_busy_cycles =
      m_cache_data_port_busy_cycles + other.m_cache_data_port_busy_cycles;
  sum.m_cache_fill_port_busy_cycles =
      m_cache_fill_port_busy_cycles + other.m_cache_fill_port_busy_cycles;
  return sum;
}

// Accumulate other's stats into this instance in-place
// Same element-wise addition as operator+, but avoid creating a temporary object
CacheStats &CacheStats::operator+=(const CacheStats &other) {
  for (int i = 0; i < NUM_MEM_ACCESS_TYPE; i++) {
    for (int j = 0; j < NUM_CACHE_REQUEST_STATUS; j++) {
      m_stats[i][j] += other.m_stats[i][j];
      m_fail_stats[i][j] += other.m_fail_stats[i][j];
    }
  }
  m_cache_port_available_cycles += other.m_cache_port_available_cycles;
  m_cache_data_port_busy_cycles += other.m_cache_data_port_busy_cycles;
  m_cache_fill_port_busy_cycles += other.m_cache_fill_port_busy_cycles;
  return *this;
}

// Return total HIT count across all access types
uint64_t CacheStats::get_hit() const {
  uint64_t hit = 0;
  for (int i = 0; i < NUM_MEM_ACCESS_TYPE; i++) {
    for (int j = 0; j < NUM_CACHE_REQUEST_STATUS; j++) {
      if (j == HIT) hit += m_stats[i][j];
    }
  }
  return hit;
}

// Return HIT + HIT_RESERVED count for GLOBAL_ACC_R (read) accesses only
// HIT_RESERVED means the line is being filled but the MSHR already has the data pending.
uint64_t CacheStats::get_read_hit() const {
  uint64_t hit = 0;
  mem_access_type types[] = {GLOBAL_ACC_R};
  CacheRequestStatus status[] = {HIT, HIT_RESERVED};
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      hit += m_stats[types[i]][status[j]];
    }
  }
  return hit;
}

// Return HIT + HIT_RESERVED count for write access types (GLOBAL_ACC_W, L2_CACHE_WA, L2_CACHE_WB)
uint64_t CacheStats::get_write_hit() const {
  uint64_t hit = 0;
  mem_access_type types[] = {GLOBAL_ACC_W, L2_CACHE_WA, L2_CACHE_WB};
  CacheRequestStatus status[] = {HIT, HIT_RESERVED};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      hit += m_stats[types[i]][status[j]];
    }
  }
  return hit;
}

// Return total MISS + SECTOR_MISS count across all access types
// SECTOR_MISS means the cache line tag matched but the specific sector was not present
uint64_t CacheStats::get_miss() const {
  uint64_t miss = 0;
  for (int i = 0; i < NUM_MEM_ACCESS_TYPE; i++) {
    for (int j = 0; j < NUM_CACHE_REQUEST_STATUS; j++) {
      if (j == MISS || j == SECTOR_MISS) miss += m_stats[i][j];
    }
  }
  return miss;
}

// Return MISS + SECTOR_MISS count for GLOBAL_ACC_R (read) accesses only
uint64_t CacheStats::get_read_miss() const {
  uint64_t miss = 0;
  mem_access_type types[] = {GLOBAL_ACC_R};
  CacheRequestStatus status[] = {MISS, SECTOR_MISS};
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      miss += m_stats[types[i]][status[j]];
    }
  }
  return miss;
}

// Return MISS + SECTOR_MISS count for write access types (GLOBAL_ACC_W, L2_CACHE_WA, L2_CACHE_WB)
uint64_t CacheStats::get_write_miss() const {
  uint64_t miss = 0;
  mem_access_type types[] = {GLOBAL_ACC_W, L2_CACHE_WA, L2_CACHE_WB};
  CacheRequestStatus status[] = {MISS, SECTOR_MISS};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      miss += m_stats[types[i]][status[j]];
    }
  }
  return miss;
}

// Return total accesses = HIT + MISS + SECTOR_MISS + HIT_RESERVED across all access types
// Exclude RESERVATION_FAIL and MSHR_HIT since those are not true new accesses
uint64_t CacheStats::get_accesses() const {
  uint64_t access = 0;
  for (int i = 0; i < NUM_MEM_ACCESS_TYPE; i++) {
    for (int j = 0; j < NUM_CACHE_REQUEST_STATUS; j++) {
      if(j == HIT || j == MISS || j == SECTOR_MISS || j == HIT_RESERVED)
        access += m_stats[i][j];
    }
  }
  return access;
}

// Return the number of hits since the last call to this function
// Snapshot the current total hit count, computes delta from m_prev_hit, and updates m_prev_hit
// Used for periodic (windowed) reporting without resetting the global counters
uint64_t CacheStats::get_interval_hit() {
  uint64_t prev_hit = m_prev_hit;
  m_prev_hit = get_hit();

  return m_prev_hit - prev_hit;
}

// Return the number of misses since the last call to this function
// Same windowed delta mechanism as get_interval_hit()
uint64_t CacheStats::get_interval_miss() {
  uint64_t prev_miss = m_prev_miss;
  m_prev_miss = get_miss();

  return m_prev_miss - prev_miss;
}

// Print a summary line (total hits, misses, hit ratio) followed by the full m_stats table broken down by [access_type][request_status], and per-type totals
// Exclude RESERVATION_FAIL and MSHR_HIT from per-type totals as they are not true accesses
void CacheStats::print_stats(FILE *out, const char *cache_name) const {
  uint64_t hit = get_hit();
  uint64_t miss = get_miss();
  fprintf(out, "\tCache Hit : %lu, Cache Miss : %lu, Hit Ratio : %.2f\n", hit,
          miss, (float)hit / (get_accesses()));
  std::vector<uint32_t> total_access;
  total_access.resize(NUM_MEM_ACCESS_TYPE, 0);
  for (int type = 0; type < NUM_MEM_ACCESS_TYPE; type++) {
    for (int status = 0; status < NUM_CACHE_REQUEST_STATUS; status++) {
      fprintf(out, "\t%s[%s][%s] = %lu\n", cache_name,
              mem_access_type_str[type], cache_request_status_str[status],
              m_stats[type][status]);
      if (status != RESERVATION_FAIL && status != MSHR_HIT)
        total_access[type] += m_stats[type][status];
    }
  }
  for (int type = 0; type < NUM_MEM_ACCESS_TYPE; type++) {
    fprintf(out, "\t%s[%s][TOTAL] = %u\n", cache_name,
            mem_access_type_str[type], total_access[type]);
  }
}

// Print the m_fail_stats table broken down by [access_type][reservation_fail_reason]
// Show why accesses were turned away (e.g. MSHR full, miss queue full, data port busy)
void CacheStats::print_fail_stats(FILE *out, const char *cache_name) const {
  for (int type = 0; type < NUM_MEM_ACCESS_TYPE; type++) {
    for (int status = 0; status < NUM_CACHE_RESERVATION_FAIL_REASON; status++) {
      fprintf(out, "\t%s[%s][%s] = %lu\n", cache_name,
              mem_access_type_str[type],
              cache_reservation_fail_reason_str[status],
              m_fail_stats[type][status]);
    }
  }
}

// Print read/write hit and miss counts in a compact format suitable for energy modeling tools
// Format: <cache_name>_RH, _RM, _WH, _WM (read hit, read miss, write hit, write miss)
void CacheStats ::print_energy_stats(FILE *out, const char *cache_name) const {
  fprintf(out, "%s_RH: %lu\n", cache_name, get_read_hit());
  fprintf(out, "%s_RM: %lu\n", cache_name, get_read_miss());
  fprintf(out, "%s_WH: %lu\n", cache_name, get_write_hit());
  fprintf(out, "%s_WM: %lu\n", cache_name, get_write_miss());
}

// Return true if access_type and access_outcome are both within their valid enum ranges
// Used as a guard assertion before indexing into m_stats
bool CacheStats::check_valid(int access_type, int access_outcome) const {
  return (access_type >= 0 && access_type < NUM_MEM_ACCESS_TYPE &&
          access_outcome >= 0 && access_outcome < NUM_CACHE_REQUEST_STATUS);
}

// Return true if access_type and fail_outcome are both within their valid enum ranges
// Used as a guard assertion before indexing into m_fail_stats
bool CacheStats::check_fail_valid(int access_type, int fail_outcome) const {
  return (access_type >= 0 && access_type < NUM_MEM_ACCESS_TYPE &&
          fail_outcome >= 0 &&
          fail_outcome < NUM_CACHE_RESERVATION_FAIL_REASON);
}
