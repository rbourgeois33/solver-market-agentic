// Shared pieces of the SMA bench drivers (header-only).
//
// Server protocol (driven by the Python dispatcher, sma/worker):
//   driver --system DIR       loads DIR/A.csr and DIR/b.vec ONCE, uploads to the GPU,
//                             prints "READY <load json>" on stdout
//   stdin : one request file path per line ("QUIT" to exit)
//   stdout: "DONE <result json path>" after each request
// Request file: "key value" lines (value = rest of the line):
//   id, out (result json path), tol (true relative residual target),
//   repeats (timed setups and solves), warmup (untimed setup+solve first),
//   plus library-specific keys (config, solver, belos_xml, muelu_xml).
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace sma {

// ------------------------------------------------------------------ binary CSR / vector
struct Csr {
  int64_t nrows = 0, ncols = 0, nnz = 0;
  std::vector<int64_t> rowptr;
  std::vector<int64_t> colind;  // widened to 64-bit on read; drivers narrow as needed
  std::vector<double> values;
};

inline void read_at(FILE* f, uint64_t off, void* dst, size_t bytes, const std::string& path) {
  if (fseeko(f, (off_t)off, SEEK_SET) != 0 || fread(dst, 1, bytes, f) != bytes)
    throw std::runtime_error("short read in " + path);
}

inline Csr read_csr(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + path);
  unsigned char h[128];
  read_at(f, 0, h, 128, path);
  if (std::memcmp(h, "SMACSR01", 8) != 0) throw std::runtime_error(path + ": bad magic");
  uint32_t version, idx_bytes;
  uint64_t nrows, ncols, nnz, o_r, o_c, o_v;
  std::memcpy(&version, h + 8, 4);
  std::memcpy(&idx_bytes, h + 12, 4);
  std::memcpy(&nrows, h + 16, 8);
  std::memcpy(&ncols, h + 24, 8);
  std::memcpy(&nnz, h + 32, 8);
  std::memcpy(&o_r, h + 40, 8);
  std::memcpy(&o_c, h + 48, 8);
  std::memcpy(&o_v, h + 56, 8);
  if (version != 1) throw std::runtime_error(path + ": unsupported version");
  Csr A;
  A.nrows = (int64_t)nrows; A.ncols = (int64_t)ncols; A.nnz = (int64_t)nnz;
  A.rowptr.resize(nrows + 1);
  A.values.resize(nnz);
  A.colind.resize(nnz);
  read_at(f, o_r, A.rowptr.data(), 8 * (nrows + 1), path);
  if (idx_bytes == 8) {
    read_at(f, o_c, A.colind.data(), 8 * nnz, path);
  } else {
    std::vector<int32_t> c32(nnz);
    read_at(f, o_c, c32.data(), 4 * nnz, path);
    std::copy(c32.begin(), c32.end(), A.colind.begin());
  }
  read_at(f, o_v, A.values.data(), 8 * nnz, path);
  fclose(f);
  return A;
}

inline std::vector<double> read_vec(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + path);
  unsigned char h[128];
  read_at(f, 0, h, 128, path);
  if (std::memcmp(h, "SMAVEC01", 8) != 0) throw std::runtime_error(path + ": bad magic");
  uint64_t n, off;
  std::memcpy(&n, h + 16, 8);
  std::memcpy(&off, h + 24, 8);
  std::vector<double> v(n);
  read_at(f, off, v.data(), 8 * n, path);
  fclose(f);
  return v;
}

// ------------------------------------------------------------------ timing
inline double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline double median(std::vector<double> v) {
  if (v.empty()) return NAN;
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// ------------------------------------------------------------------ GPU memory sampler
// Polls cudaMemGetInfo every ~1 ms on a background thread: peak device memory in use
// (all processes on this GPU; the dispatcher runs one benchmark at a time).
class MemSampler {
 public:
  MemSampler() { baseline_ = used(); }
  static size_t used() {
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    return total_b - free_b;
  }
  static size_t total() {
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    return total_b;
  }
  void start() {
    baseline_ = used();
    peak_ = baseline_;
    run_ = true;
    th_ = std::thread([this] {
      while (run_) {
        size_t u = used();
        size_t p = peak_.load();
        while (u > p && !peak_.compare_exchange_weak(p, u)) {}
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
      }
    });
  }
  void stop() {
    run_ = false;
    if (th_.joinable()) th_.join();
    size_t u = used();
    if (u > peak_) peak_ = u;
  }
  size_t baseline() const { return baseline_; }
  size_t peak() const { return peak_; }

 private:
  std::atomic<bool> run_{false};
  std::atomic<size_t> peak_{0};
  size_t baseline_ = 0;
  std::thread th_;
};

// ------------------------------------------------------------------ true residual (host, independent of libs)
inline double norm2(const std::vector<double>& v) {
  long double s = 0;
  for (double x : v) s += (long double)x * x;
  return std::sqrt((double)s);
}

inline double true_relres(const Csr& A, const std::vector<double>& x, const std::vector<double>& b) {
  long double s = 0;
#pragma omp parallel for reduction(+ : s)
  for (int64_t i = 0; i < A.nrows; ++i) {
    double r = b[i];
    for (int64_t k = A.rowptr[i]; k < A.rowptr[i + 1]; ++k) r -= A.values[k] * x[A.colind[k]];
    s += (long double)r * r;
  }
  double nb = norm2(b);
  return std::sqrt((double)s) / (nb > 0 ? nb : 1.0);
}

// ------------------------------------------------------------------ request / result
struct Request {
  std::map<std::string, std::string> kv;
  std::string get(const std::string& k, const std::string& def = "") const {
    auto it = kv.find(k);
    return it == kv.end() ? def : it->second;
  }
  std::string need(const std::string& k) const {
    auto it = kv.find(k);
    if (it == kv.end()) throw std::runtime_error("request misses key '" + k + "'");
    return it->second;
  }
};

inline Request read_request(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open request " + path);
  Request r;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    auto sp = line.find(' ');
    if (sp == std::string::npos) r.kv[line] = "";
    else r.kv[line.substr(0, sp)] = line.substr(sp + 1);
  }
  return r;
}

// "key value" config lines grouped by first word (used by the hypre driver).
struct ConfigLine {
  std::string section, key, value;
};

inline std::vector<ConfigLine> read_config_lines(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::vector<ConfigLine> out;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    ConfigLine c;
    ss >> c.section;
    if (c.section == "solver") { ss >> c.value; out.push_back(c); continue; }
    ss >> c.key >> c.value;
    out.push_back(c);
  }
  return out;
}

inline std::string jstr(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += c; }
    else if (c == '\n') o += "\\n";
    else if ((unsigned char)c < 0x20) o += ' ';
    else o += c;
  }
  return o + "\"";
}

inline std::string jnum(double v) {
  if (!std::isfinite(v)) return "null";
  char b[64];
  snprintf(b, sizeof b, "%.9g", v);
  return b;
}

// Minimal ordered JSON object writer.
class Json {
 public:
  Json& num(const std::string& k, double v) { return raw(k, jnum(v)); }
  Json& integer(const std::string& k, long long v) { return raw(k, std::to_string(v)); }
  Json& str(const std::string& k, const std::string& v) { return raw(k, jstr(v)); }
  Json& boolean(const std::string& k, bool v) { return raw(k, v ? "true" : "false"); }
  Json& vec(const std::string& k, const std::vector<double>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + jnum(v[i]);
    return raw(k, s + "]");
  }
  Json& raw(const std::string& k, const std::string& v) {
    items_.emplace_back(k, v);
    return *this;
  }
  std::string dump() const {
    std::string s = "{\n";
    for (size_t i = 0; i < items_.size(); ++i)
      s += "  " + jstr(items_[i].first) + ": " + items_[i].second + (i + 1 < items_.size() ? ",\n" : "\n");
    return s + "}\n";
  }
  void write(const std::string& path) const {
    std::string tmp = path + ".tmp";
    { std::ofstream o(tmp); o << dump(); }
    std::rename(tmp.c_str(), path.c_str());  // atomic for readers polling the file
  }

 private:
  std::vector<std::pair<std::string, std::string>> items_;
};

inline std::string gpu_name() {
  cudaDeviceProp p;
  int dev = 0;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&p, dev);
  return p.name;
}

// Common result fields filled by every driver.
struct RunStats {
  std::vector<double> t_setup, t_solve;
  std::vector<double> relres_true;
  int iters = -1;
  bool lib_converged = false;
  size_t mem_baseline = 0, mem_peak = 0;
  double op_complexity = NAN, grid_complexity = NAN;
  int levels = -1;
  std::string error;
};

inline void fill_result(Json& j, const Request& rq, const std::string& lib, const std::string& lib_version,
                        const RunStats& s, double tol) {
  double rr = s.relres_true.empty() ? NAN : *std::max_element(s.relres_true.begin(), s.relres_true.end());
  bool ok = s.error.empty() && std::isfinite(rr) && rr <= tol;
  j.str("id", rq.get("id")).str("lib", lib).str("lib_version", lib_version).str("gpu", gpu_name());
  j.boolean("converged", ok).boolean("lib_converged", s.lib_converged).integer("iterations", s.iters);
  j.num("true_relres", rr).num("tol", tol);
  j.num("t_setup_median", median(s.t_setup)).num("t_solve_median", median(s.t_solve));
  j.vec("t_setup", s.t_setup).vec("t_solve", s.t_solve);
  j.integer("gpu_mem_baseline_bytes", (long long)s.mem_baseline);
  j.integer("gpu_mem_peak_bytes", (long long)s.mem_peak);
  j.integer("gpu_mem_peak_delta_bytes", (long long)(s.mem_peak - std::min(s.mem_peak, s.mem_baseline)));
  j.integer("gpu_mem_total_bytes", (long long)MemSampler::total());
  j.num("operator_complexity", s.op_complexity).num("grid_complexity", s.grid_complexity);
  j.integer("levels", s.levels);
  j.str("error", s.error);
}

// ------------------------------------------------------------------ server loop
// Each driver provides: load(Csr, b) once, then run(Request) -> Json for each request.
template <class Driver>
int serve(Driver& d, int argc, char** argv) {
  std::string system;
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--system" && i + 1 < argc) system = argv[++i];
  if (system.empty()) { fprintf(stderr, "usage: %s --system DIR\n", argv[0]); return 2; }
  double t0 = now();
  Csr A = read_csr(system + "/A.csr");
  std::vector<double> b = read_vec(system + "/b.vec");
  double t_read = now() - t0;
  t0 = now();
  d.load(A, b);
  double t_upload = now() - t0;
  printf("READY {\"t_read\": %s, \"t_upload\": %s, \"nrows\": %lld, \"nnz\": %lld}\n", jnum(t_read).c_str(),
         jnum(t_upload).c_str(), (long long)A.nrows, (long long)A.nnz);
  fflush(stdout);
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line == "QUIT") break;
    if (line.empty()) continue;
    Request rq;
    std::string out;
    try {
      rq = read_request(line);
      out = rq.need("out");
      Json j = d.run(rq, A, b);
      j.write(out);
    } catch (std::exception& e) {
      // Errors must not kill the server (the matrix stays loaded): report and continue.
      Json j;
      j.str("id", rq.get("id")).boolean("converged", false).str("error", e.what());
      if (out.empty()) out = line + ".result.json";
      j.write(out);
    }
    printf("DONE %s\n", out.c_str());
    fflush(stdout);
  }
  return 0;
}

}  // namespace sma
