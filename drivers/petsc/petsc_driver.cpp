// PETSc KSP + PCGAMG bench server (GPU matrices: aijkokkos / aijcusparse, 1 MPI rank in v1).
// Config: one line of PETSc options (written by sma/deck.py), inserted into the global options
// database for the duration of one request. mat_type/vec_type choose the GPU backend
// (default aijkokkos/kokkos); one matrix per mat_type is cached so the CSR is uploaded once per type.
#include <petscksp.h>

#include "sma_common.hpp"

using namespace sma;

namespace {

void ck(PetscErrorCode e, const char* what) {
  if (e) throw std::runtime_error(std::string("PETSc error in ") + what + " (code " + std::to_string(e) + ")");
}

struct PetscDriver {
  std::map<std::string, Mat> mats;  // by mat_type
  std::vector<PetscInt> coo_i, coo_j;
  std::vector<PetscScalar> vals, hb;
  PetscInt n = 0;

  void load(const Csr& h, const std::vector<double>& b) {
    n = (PetscInt)h.nrows;
    coo_i.resize(h.nnz);
    coo_j.resize(h.nnz);
    vals.assign(h.values.begin(), h.values.end());
    for (int64_t i = 0; i < h.nrows; ++i)
      for (int64_t k = h.rowptr[i]; k < h.rowptr[i + 1]; ++k) { coo_i[k] = (PetscInt)i; coo_j[k] = (PetscInt)h.colind[k]; }
    hb.assign(b.begin(), b.end());
    get_mat("aijkokkos");  // default backend uploaded at load time
  }

  Mat get_mat(const std::string& type) {
    auto it = mats.find(type);
    if (it != mats.end()) return it->second;
    Mat A;
    ck(MatCreate(PETSC_COMM_WORLD, &A), "MatCreate");
    ck(MatSetSizes(A, n, n, PETSC_DETERMINE, PETSC_DETERMINE), "MatSetSizes");
    ck(MatSetType(A, type.c_str()), "MatSetType");
    ck(MatSetPreallocationCOO(A, (PetscCount)coo_i.size(), coo_i.data(), coo_j.data()), "MatSetPreallocationCOO");
    ck(MatSetValuesCOO(A, vals.data(), INSERT_VALUES), "MatSetValuesCOO");
    ck(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY), "MatAssemblyBegin");
    ck(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY), "MatAssemblyEnd");
    mats[type] = A;
    return A;
  }

  static std::string opt_string(const char* name, const char* def) {
    char buf[256];
    PetscBool set = PETSC_FALSE;
    PetscOptionsGetString(nullptr, nullptr, name, buf, sizeof buf, &set);
    return set ? std::string(buf) : std::string(def);
  }

  Json run(const Request& rq, const Csr& hA, const std::vector<double>& hbv) {
    double tol = std::stod(rq.need("tol"));
    int repeats = std::stoi(rq.get("repeats", "3"));
    int warmup = std::stoi(rq.get("warmup", "1"));
    std::ifstream in(rq.need("config"));
    std::string opts((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    ck(PetscOptionsClear(nullptr), "PetscOptionsClear");
    ck(PetscOptionsInsertString(nullptr, opts.c_str()), "PetscOptionsInsertString");
    std::string mat_type = opt_string("-mat_type", "aijkokkos");
    std::string vec_type = opt_string("-vec_type", "");
    Mat A = get_mat(mat_type);
    Vec x, b;
    ck(MatCreateVecs(A, &x, &b), "MatCreateVecs");
    if (!vec_type.empty()) { ck(VecSetType(x, vec_type.c_str()), "VecSetType"); ck(VecSetType(b, vec_type.c_str()), "VecSetType"); }
    {
      PetscScalar* p;
      ck(VecGetArrayWrite(b, &p), "VecGetArrayWrite");
      std::copy(hb.begin(), hb.end(), p);
      ck(VecRestoreArrayWrite(b, &p), "VecRestoreArrayWrite");
    }

    RunStats st;
    std::string ksp_type, pc_type, reason_str;
    MemSampler mem;
    mem.start();
    st.mem_baseline = mem.baseline();
    try {
      for (int r = -warmup; r < repeats; ++r) {
        KSP ksp;
        ck(KSPCreate(PETSC_COMM_WORLD, &ksp), "KSPCreate");
        ck(KSPSetOperators(ksp, A, A), "KSPSetOperators");
        ck(KSPSetTolerances(ksp, tol, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT), "KSPSetTolerances");
        ck(KSPSetFromOptions(ksp), "KSPSetFromOptions");  // options override the default rtol
        ck(VecSet(x, 0.0), "VecSet");
        ck(PetscDeviceContextSynchronize(nullptr), "sync");
        double t0 = now();
        ck(KSPSetUp(ksp), "KSPSetUp");
        PC pc;
        ck(KSPGetPC(ksp, &pc), "KSPGetPC");
        ck(PCSetUp(pc), "PCSetUp");
        ck(PetscDeviceContextSynchronize(nullptr), "sync");
        double ts = now() - t0;
        t0 = now();
        ck(KSPSolve(ksp, b, x), "KSPSolve");
        ck(PetscDeviceContextSynchronize(nullptr), "sync");
        double tv = now() - t0;
        if (r >= 0) {
          st.t_setup.push_back(ts);
          st.t_solve.push_back(tv);
          PetscInt it;
          KSPGetIterationNumber(ksp, &it);
          st.iters = (int)it;
          KSPConvergedReason reason;
          KSPGetConvergedReason(ksp, &reason);
          st.lib_converged = reason > 0;
          reason_str = KSPConvergedReasons[reason];
          const PetscScalar* px;
          ck(VecGetArrayRead(x, &px), "VecGetArrayRead");
          std::vector<double> hx(px, px + n);
          ck(VecRestoreArrayRead(x, &px), "VecRestoreArrayRead");
          st.relres_true.push_back(true_relres(hA, hx, hbv));
          if (r == 0) {
            KSPType kt;
            PCType pt;
            KSPGetType(ksp, &kt);
            PCGetType(pc, &pt);
            ksp_type = kt;
            pc_type = pt;
            PetscBool is_mg = PETSC_FALSE;
            PetscObjectTypeCompareAny((PetscObject)pc, &is_mg, PCGAMG, PCMG, "");
            if (is_mg) {
              PetscInt L;
              PCMGGetLevels(pc, &L);
              st.levels = (int)L;
              PetscReal gc, oc;
              if (PCMGGetGridComplexity(pc, &gc, &oc) == 0) { st.grid_complexity = gc; st.op_complexity = oc; }
            }
          }
        }
        KSPDestroy(&ksp);
      }
    } catch (std::exception& e) {
      st.error = e.what();
    }
    mem.stop();
    st.mem_peak = mem.peak();

    // Options given but never queried: typos or options that do not apply (reported to the agent).
    PetscInt nleft;
    char** names;
    char** values;
    std::string unused;
    if (PetscOptionsLeftGet(nullptr, &nleft, &names, &values) == 0) {
      for (PetscInt i = 0; i < nleft; ++i) unused += (unused.empty() ? "-" : " -") + std::string(names[i]);
      PetscOptionsLeftRestore(nullptr, &nleft, &names, &values);
    }
    VecDestroy(&x);
    VecDestroy(&b);
    ck(PetscOptionsClear(nullptr), "PetscOptionsClear");

    char ver[256];
    PetscGetVersion(ver, sizeof ver);
    Json j;
    fill_result(j, rq, "petsc", ver, st, tol);
    j.str("ksp_type", ksp_type).str("pc_type", pc_type).str("mat_type", mat_type);
    j.str("converged_reason", reason_str).str("unused_options", unused);
    return j;
  }
};

}  // namespace

int main(int argc, char** argv) {
  // No PETSc options on the command line: each request brings its own.
  int pargc = 1;
  if (PetscInitialize(&pargc, &argv, nullptr, nullptr)) return 1;
  int rc;
  {
    PetscDriver d;
    rc = serve(d, argc, argv);
    for (auto& kv : d.mats) MatDestroy(&kv.second);
  }
  PetscFinalize();
  return rc;
}
