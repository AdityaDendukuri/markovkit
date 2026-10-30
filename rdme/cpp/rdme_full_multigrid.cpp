#include "core/types.hpp"
#include "cascading_smoother.hpp"
#include "block_cascading_smoother.hpp"
#include "linear/solvers/gmres.hpp"
#include "linear/solvers/ilu.hpp"
#include "linear/solvers/auto_linear.hpp"
#include <sys/resource.h>
#include <klu.h>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// Single-species, total-population FSP on a reflecting Cartesian voxel grid.
// All levels contain the complete reaction-diffusion operator. Nothing is
// dropped from the fine generator, including birth losses at the FSP boundary.
namespace {
using Vec = num::vec<double>;
using Index = std::uint32_t;
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
struct ProfileTimer {
  double *total;
  Clock::time_point start;
  ProfileTimer(bool enabled,double &value):total(enabled?&value:nullptr) {
    if(total)start=Clock::now();
  }
  ~ProfileTimer(){if(total)*total+=seconds(start);}
};
double norm1(const Vec &v) {
  double sum = 0.;
  for (double x : v) sum += std::abs(x);
  return sum;
}
struct Parameters {
  int dimension = 1, cells = 4, cap = 40, sweeps = 2, steps = 1;
  int max_iterations = 1000, restart = 30, initial = 8, cycles = 1;
  double diffusion = 1., shift = .05, tolerance = 1e-9;
  int cascade_depth = 2, cascade_row_cap = 0;
  double cascade_rcond = 1e-8;
  double uniformization_mean = 8.;
  double fsp_tolerance = 1e-6;
  double c1 = .0133, c2 = .0006, c3 = .6924125, c4 = .114725;
  std::string method = "mg", smoother = "gs", integrator = "be", hierarchy = "recursive", output, history, state_output;
  // "standard": p.cycles plain recursive coarse visits (V or W). "k": K-cycle,
  // two GCR iterations on each coarse system, preconditioned by the next cycle.
  std::string cycle_type = "standard", cascade_order, block_kernel="structured", transfer_output;
  bool export_only=false;
  bool check = false, adaptive_fsp = false, reuse_fsp = true, profile=false, reflection=false;
};

// Lexicographic weak-composition rank, with increasing total population.
// Optional central reflection v -> V-1-v stores total probability on each
// two-state orbit. Exports lift this exactly to the original state ordering.
// The precomputed binomial table avoids factorials and per-state hash tables.
struct Space {
  int voxels, cap;
  std::vector<std::vector<std::uint64_t>> choose;
  std::uint64_t size, full_size;
  bool reflected=false;
  std::vector<Index> folded, offsets;
  std::vector<unsigned char> orbit_size;
  Space(int v, int k, bool reflection=false) : voxels(v), cap(k),
      choose(k + v + 1, std::vector<std::uint64_t>(v + 1, 0)) {
    constexpr auto limit = std::numeric_limits<std::uint64_t>::max();
    for (int n = 0; n <= k + v; ++n) {
      choose[n][0] = 1;
      for (int r = 1; r <= std::min(n, v); ++r) {
        const auto a = choose[n-1][r-1], b = choose[n-1][r];
        choose[n][r] = a > limit-b ? limit : a+b;
      }
    }
    size = choose[k+v][v];
    if (size > 8000000)
      throw std::runtime_error("FSP exceeds this experiment's 8-million-state safety limit");
    full_size=size;
    if(reflection && v>1) {
      folded.assign(full_size,std::numeric_limits<Index>::max());
      offsets.resize(cap+2);
      std::vector<int> reversed(v);
      int previous_total=-1;
      each([&](Index raw,const std::vector<int> &n,int total) {
        if(total!=previous_total) {offsets[total]=orbit_size.size();previous_total=total;}
        if(folded[raw]!=std::numeric_limits<Index>::max())return;
        std::reverse_copy(n.begin(),n.end(),reversed.begin());
        const Index mirror=raw_rank(reversed,total), next=orbit_size.size();
        folded[raw]=folded[mirror]=next;
        orbit_size.push_back(raw==mirror?1:2);
      });
      size=orbit_size.size(); offsets[cap+1]=size;
      reflected=true;
    }
  }
  Index raw_rank(const std::vector<int> &n, int total) const {
    std::uint64_t result = total == 0 ? 0 : choose[total+voxels-1][voxels];
    int remaining = total;
    for (int v = 0; v+1 < voxels; ++v) {
      const int rest = voxels-v-1;
      result += choose[remaining+rest][rest] - choose[remaining-n[v]+rest][rest];
      remaining -= n[v];
    }
    return static_cast<Index>(result);
  }
  Index rank(const std::vector<int> &n,int total) const {
    const Index raw=raw_rank(n,total);
    return reflected?folded[raw]:raw;
  }
  Index offset(int total) const {
    return reflected?offsets[total]:(total==0?0:choose[total+voxels-1][voxels]);
  }
  int multiplicity(Index state) const {return reflected?orbit_size[state]:1;}
  void write_state(const Vec &probability,std::ostream &out) const {
    out<<std::setprecision(17)<<"state,probability\n";
    for(Index raw=0;raw<full_size;++raw) {
      const Index state=reflected?folded[raw]:raw;
      out<<raw<<','<<probability[state]/multiplicity(state)<<'\n';
    }
  }
  template<class Function> void each(Function function, int first_total=0) const {
    std::vector<int> counts(voxels, 0);
    Index index=offset(first_total);
    Index raw=first_total==0?0:choose[first_total+voxels-1][voxels];
    std::function<void(int,int,int)> visit = [&](int v,int remaining,int total) {
      if (v+1 == voxels) {
        counts[v] = remaining;
        if(!reflected || folded[raw]==index)function(index++,counts,total);
        ++raw;
      } else {
        for (int n = 0; n <= remaining; ++n) {
          counts[v] = n;
          visit(v+1, remaining-n, total);
        }
      }
    };
    for (int k = first_total; k <= cap; ++k) visit(0,k,k);
    if (index != size) throw std::logic_error("enumeration size mismatch");
  }
};

struct Matrix {
  using domain_type = Vec;
  using codomain_type = Vec;
  std::vector<Index> ptr{0}, column, diagonal;
  std::vector<double> value;
  num::idx rows() const { return diagonal.size(); }
  num::idx cols() const { return rows(); }
  void apply(const Vec &x, Vec &y) const {
    for (Index row = 0; row < rows(); ++row) {
      double sum = 0.;
      for (Index j = ptr[row]; j < ptr[row+1]; ++j)
        sum += value[j]*x[column[j]];
      y[row] = sum;
    }
  }
  double residual_norm(const Vec &b,const Vec &x,Vec &r) const {
    double norm=0.;
    for(Index row=0;row<rows();++row) {
      double sum=0.;
      for(Index j=ptr[row];j<ptr[row+1];++j)sum+=value[j]*x[column[j]];
      r[row]=b[row]-sum; norm+=std::abs(r[row]);
    }
    return norm;
  }
  void smooth(const Vec &b, Vec &x, int sweeps, bool reverse,
              bool jacobi, Vec &scratch) const {
    for (int sweep = 0; sweep < sweeps; ++sweep) {
      if (jacobi) {
        apply(x,scratch);
        for (Index row = 0; row < rows(); ++row)
          x[row] += (2./3.)*(b[row]-scratch[row])/value[diagonal[row]];
      } else {
        for (Index i = 0; i < rows(); ++i) {
          const Index row = reverse ? rows()-1-i : i;
          double sum = b[row];
          for (Index j = ptr[row]; j < ptr[row+1]; ++j)
            if (column[j] != row) sum -= value[j]*x[column[j]];
          x[row] = sum/value[diagonal[row]];
        }
      }
    }
  }
  std::uint64_t bytes() const {
    return 4*(ptr.size()+column.size()+diagonal.size())+8*value.size();
  }
};

// Population blocks fit the driver's 32-bit state indices. Keep that compact
// pattern through factorization instead of copying it into a generic 64-bit
// sparse matrix. The existing numerics ILU kernels do the same arithmetic.
struct PopulationBlockILU {
  Matrix factor;
  std::vector<Index> lower_begin, upper_end;
  std::vector<double> pivot;
  PopulationBlockILU(const Matrix &matrix, Index first, Index last, bool structured) {
    const Index n=last-first;
    if(structured) {
      lower_begin.resize(n); upper_end.resize(n); pivot.resize(n);
      for(Index row=first;row<last;++row) {
        const Index i=row-first, diagonal=matrix.diagonal[row];
        lower_begin[i]=std::lower_bound(matrix.column.begin()+matrix.ptr[row],
                         matrix.column.begin()+diagonal,first)-matrix.column.begin()-matrix.ptr[row];
        upper_end[i]=std::lower_bound(matrix.column.begin()+diagonal+1,
                         matrix.column.begin()+matrix.ptr[row+1],last)-matrix.column.begin()-matrix.ptr[row];
        double d=matrix.value[diagonal];
        // Cartesian diffusion changes the checkerboard parity of an arrangement.
        // The block graph is bipartite: no elimination update can hit a stored
        // off-diagonal entry. ILU(0) therefore modifies only the diagonal.
        for(Index j=matrix.ptr[row]+lower_begin[i];j<diagonal;++j) {
          const Index column=matrix.column[j];
          auto begin=matrix.column.begin()+matrix.diagonal[column]+1;
          auto end=matrix.column.begin()+matrix.ptr[column+1];
          auto transpose=std::lower_bound(begin,end,row);
          if(transpose==end || *transpose!=row)
            throw std::logic_error("diffusion block has no reverse edge");
          const double multiplier=matrix.value[j]/pivot[column-first];
          d-=multiplier*matrix.value[transpose-matrix.column.begin()];
        }
        if(!(d>0.) || !std::isfinite(d))
          throw std::runtime_error("structured population-block ILU invalid pivot");
        pivot[i]=d;
      }
      // The triangular sweeps need reciprocals only. Form these once rather
      // than dividing twice per state on every block visit.
      for(double &d:pivot)d=1./d;
      return;
    }
    std::size_t entries=0;
    for(Index row=first;row<last;++row)
      for(Index j=matrix.ptr[row];j<matrix.ptr[row+1];++j)
        if(matrix.column[j]>=first && matrix.column[j]<last)++entries;
    factor.column.reserve(entries); factor.value.reserve(entries);
    factor.ptr.reserve(n+1); factor.diagonal.resize(n);
    for(Index row=first;row<last;++row) {
      for(Index j=matrix.ptr[row];j<matrix.ptr[row+1];++j) {
        const Index column=matrix.column[j];
        if(column<first || column>=last)continue;
        if(column==row)factor.diagonal[row-first]=factor.value.size();
        factor.column.push_back(column-first); factor.value.push_back(matrix.value[j]);
      }
      factor.ptr.push_back(factor.value.size());
    }
    std::vector<Index> workspace(n);
    if(!num::ilu0_factor(factor.value.data(),factor.ptr.data(),factor.column.data(),
                        factor.diagonal.data(),workspace.data(),n))
      throw std::runtime_error("population-block ILU zero pivot");
  }
  void apply(double *work) const {
    num::csr_lu_solve(work,factor.value.data(),factor.ptr.data(),factor.column.data(),
                      factor.diagonal.data(),work,static_cast<Index>(factor.rows()));
  }
  void smooth(const Matrix &matrix,const Vec &b,Vec &x,Index first,double *work) const {
    // Forward substitution in (D+L), fused with r=b-Ax. Original matrix
    // entries serve both operations; no copied sparse factor is traversed.
    for(Index i=0;i<pivot.size();++i) {
      const Index row=first+i, diagonal=matrix.diagonal[row];
      double sum=b[row];
      const Index lower=matrix.ptr[row]+lower_begin[i];
      for(Index j=matrix.ptr[row];j<lower;++j)
        sum-=matrix.value[j]*x[matrix.column[j]];
      for(Index j=lower;j<diagonal;++j) {
        const Index column=matrix.column[j];
        sum-=matrix.value[j]*(x[column]+work[column-first]);
      }
      for(Index j=diagonal;j<matrix.ptr[row+1];++j)
        sum-=matrix.value[j]*x[matrix.column[j]];
      work[i]=sum*pivot[i];
    }
    // Back substitution in (D+U); work holds D^-1 times the forward result.
    // Update x here rather than making another full vector pass afterward.
    for(Index i=static_cast<Index>(pivot.size());i-->0;) {
      const Index row=first+i;
      double sum=0.;
      for(Index j=matrix.diagonal[row]+1;j<matrix.ptr[row]+upper_end[i];++j)
        sum+=matrix.value[j]*work[matrix.column[j]-first];
      work[i]-=sum*pivot[i];
      x[row]+=work[i];
    }
  }
  std::uint64_t bytes() const {
    return factor.bytes()+4*(lower_begin.size()+upper_end.size())+8*pivot.size();
  }
};

struct Level {
  Parameters p;
  Space space;
  Matrix A;
  rdme::CascadingSmoother cascade;
  rdme::BlockCascadingSmoother block_cascade;
  double cascade_setup = 0.;
  std::unique_ptr<Level> coarse;
  std::vector<Index> parent;
  std::vector<double> weight, lower, upper, pivot;
  std::vector<Index> block_offsets;
  std::vector<std::unique_ptr<num::auto_linear_solver>> block_factors;
  std::vector<std::unique_ptr<PopulationBlockILU>> block_ilu;
  mutable std::vector<Vec> block_work, block_solution;
  mutable std::vector<double> block_ilu_work;
  mutable double smooth_seconds=0., matvec_seconds=0., transfer_seconds=0., residual_seconds=0.;
  mutable std::uint64_t cycle_calls=0;
  std::uint64_t reused_rows=0, reused_factors=0, reused_transfers=0;
  mutable Vec residual, correction, coarse_rhs, coarse_solution, scratch;
  mutable Vec krylov_v1, krylov_w1, krylov_v2, krylov_w2, krylov_r;
  // Overlapping pair-chain smoother. For a voxel edge (a,b), a chain contains
  // the states that agree outside {a,b} and have the same n_a+n_b=m>=1,
  // ordered by n_a=0,...,m. Its principal submatrix is tridiagonal: reactions
  // change the population and every other diffusion jump changes a count
  // outside the pair. Chains are stored edge by edge with precomputed Thomas
  // coefficients. States in no such chain receive a point update.
  std::vector<Index> chain_ptr, chain_state, chain_uncovered;
  std::vector<double> chain_lower, chain_inv_pivot, chain_upper_ratio;
  mutable Vec chain_work;
  // Experimental graph-neighborhood smoother. The patch at voxel v contains
  // v and its neighbors. It fixes outside counts and the total in the patch.
  // Sparse local ILU(0) factors are stored in one packed CSR array.
  std::vector<Index> overlap_ptr, overlap_state, overlap_rowptr, overlap_column,
      overlap_diag, overlap_uncovered;
  std::vector<double> overlap_factor;
  mutable Vec overlap_work;
  std::vector<unsigned short> population;
  std::vector<double> leakage;
  static int voxel_count(const Parameters &p) {
    int result=1;
    for (int d=0; d<p.dimension; ++d) {
      if(p.cells>4096/result)throw std::invalid_argument("voxel count exceeds experiment limit");
      result*=p.cells;
    }
    return result;
  }
  static bool supports_population_reuse(const Parameters &p) {
    return p.smoother=="gs" || p.smoother=="jacobi" ||
           p.smoother=="block" || p.smoother=="block-ilu";
  }
  explicit Level(Parameters parameters, bool hierarchy=true, Level *previous=nullptr)
      : p(parameters), space(voxel_count(p),p.cap,p.reflection),
        population(previous?0:space.size), leakage(previous?0:space.size,0.) {
    const int V=space.voxels;
    // Only population-independent setup is eligible for incremental growth.
    // Experimental patch/cascade smoothers retain their full-rebuild path.
    if(previous && (p.cap<=previous->p.cap || V!=previous->space.voxels ||
        p.shift!=previous->p.shift || p.diffusion!=previous->p.diffusion ||
        p.c1!=previous->p.c1 || p.c2!=previous->p.c2 ||
        p.c3!=previous->p.c3 || p.c4!=previous->p.c4 ||
        p.smoother!=previous->p.smoother || p.hierarchy!=previous->p.hierarchy ||
        p.block_kernel!=previous->p.block_kernel || p.reflection!=previous->p.reflection ||
        p.dimension!=previous->p.dimension))
      throw std::logic_error("incompatible incremental FSP expansion");
    if(previous && !supports_population_reuse(p))
      throw std::logic_error("incremental FSP expansion is unavailable for this smoother");
    const int old_cap=previous?previous->p.cap:-1;
    const Index old_size=previous?previous->rows():0;
    if(previous) {
      A=std::move(previous->A);
      population=std::move(previous->population); population.resize(space.size);
      leakage=std::move(previous->leakage); leakage.resize(space.size,0.);
      // Only the old boundary rows gain incoming death columns. Outward birth
      // losses were already retained on their diagonals by the absorbing FSP.
      reused_rows=space.offset(old_cap);
      A.column.resize(A.ptr[reused_rows]); A.value.resize(A.ptr[reused_rows]);
      A.ptr.resize(reused_rows+1);
      // Drop old workspaces before allocating the next coarse level.
      previous->residual=Vec(); previous->correction=Vec(); previous->scratch=Vec();
      previous->coarse_rhs=Vec(); previous->coarse_solution=Vec();
      previous->krylov_v1=Vec(); previous->krylov_w1=Vec();
      previous->krylov_v2=Vec(); previous->krylov_w2=Vec(); previous->krylov_r=Vec();
      smooth_seconds=previous->smooth_seconds; matvec_seconds=previous->matvec_seconds;
      transfer_seconds=previous->transfer_seconds; cycle_calls=previous->cycle_calls;
      residual_seconds=previous->residual_seconds;
    }
    residual=Vec(space.size,0.); correction=Vec(space.size,0.); scratch=Vec(space.size,0.);
    // Use C(V+K-1,V), the count with a specified molecule removed. The
    // binomial table stores columns through V, also when K is larger than V.
    const std::uint64_t eligible=p.cap>0?space.choose[V+p.cap-1][V]:0;
    if (space.size+static_cast<std::uint64_t>((2+2*p.dimension)*V)*eligible>200000000)
      throw std::runtime_error("conservative sparse-storage estimate exceeds safety limit");
    std::vector<std::vector<int>> neighbors(V);
    for (int v=0; v<V; ++v) {
      int stride=1;
      for (int d=0; d<p.dimension; ++d) {
        const int coordinate=(v/stride)%p.cells;
        if (coordinate>0) neighbors[v].push_back(v-stride);
        if (coordinate+1<p.cells) neighbors[v].push_back(v+stride);
        stride*=p.cells;
      }
    }
    const double volume=1./V, jump=p.diffusion*p.cells*p.cells;
    auto birth=[&](double n) {return p.c3*volume+p.c1*n*(n-1)/(2*volume);};
    auto death=[&](double n) {return p.c4*n+p.c2*n*(n-1)*(n-2)/(6*volume*volume);};
    // Count stored entries before appending. Reserving once avoids geometric
    // CSR growth and its large temporary old/new allocation at each expansion.
    auto shifted_states=[&](int removed)->std::uint64_t {
      return p.cap<removed?0:space.choose[p.cap-removed+V][V];
    };
    std::uint64_t directed_edges=0;
    for(const auto &adjacent:neighbors)directed_edges+=adjacent.size();
    const auto birth_entries=p.c3>0?shifted_states(1):(p.c1>0?shifted_states(3):0);
    const auto death_entries=p.c4>0?shifted_states(1):(p.c2>0?shifted_states(3):0);
    const std::uint64_t stored_entries=space.size+directed_edges*eligible+
                                      V*(birth_entries+death_entries);
    A.column.reserve(stored_entries); A.value.reserve(stored_entries);
    A.diagonal.resize(space.size);
    A.ptr.reserve(space.size+1);
    std::vector<std::pair<Index,double>> entries;
    space.each([&](Index row, std::vector<int> &n, int total) {
      if (space.rank(n,total)!=row) throw std::logic_error("rank mismatch");
      population[row]=total;
      leakage[row]=0.;
      entries.clear();
      double diag=p.shift;
      auto incoming_entry=[&](Index column,double value) {
        if(space.reflected)value*=double(space.multiplicity(row))/space.multiplicity(column);
        entries.emplace_back(column,value);
      };
      for (int v=0; v<V; ++v) {
        const int count=n[v];
        diag+=birth(count)+death(count)+jump*count*neighbors[v].size();
        if (total==p.cap) leakage[row]+=birth(count);
        if (count>0) {
          --n[v];
          const double incoming=birth(count-1);
          if (incoming>0) incoming_entry(space.rank(n,total-1),-incoming);
          for (int w:neighbors[v]) {
            ++n[w];
            incoming_entry(space.rank(n,total),-jump*n[w]);
            --n[w];
          }
          ++n[v];
        }
        if (total<p.cap) {
          ++n[v];
          const double incoming=death(count+1);
          if (incoming>0) incoming_entry(space.rank(n,total+1),-incoming);
          --n[v];
        }
      }
      entries.emplace_back(row,diag);
      std::sort(entries.begin(),entries.end());
      for (auto [column,value]:entries) {
        if(space.reflected && A.column.size()>A.ptr.back() && A.column.back()==column) {
          A.value.back()+=value;
          continue;
        }
        if (column==row) A.diagonal[row]=A.value.size();
        A.column.push_back(column); A.value.push_back(value);
      }
      if (A.value.size()>std::numeric_limits<Index>::max())
        throw std::runtime_error("CSR index overflow");
      A.ptr.push_back(A.value.size());
    }, std::max(0,old_cap));
    if(p.check && !space.reflected && A.value.size()!=stored_entries)
      throw std::logic_error("population-block sparse entry count mismatch");
    if ((p.smoother=="block" || p.smoother=="block-ilu" || p.smoother=="block-cascade") && p.cells>1) {
      block_offsets.resize(p.cap+2);
      block_offsets[0]=0;
      for (int k=1;k<=p.cap;++k)
        block_offsets[k]=space.offset(k);
      block_offsets[p.cap+1]=static_cast<Index>(space.size);
      if(previous) {
        block_factors=std::move(previous->block_factors);
        block_ilu=std::move(previous->block_ilu);
        block_work=std::move(previous->block_work);
        block_solution=std::move(previous->block_solution);
        block_ilu_work=std::move(previous->block_ilu_work);
        reused_factors=block_factors.size()+block_ilu.size();
      }
      if (p.smoother!="block-cascade") for (int k=static_cast<int>(reused_factors);k<=p.cap;++k) {
        const Index first=block_offsets[k], last=block_offsets[k+1];
        if (p.smoother=="block" && last-first>5000)
          throw std::runtime_error("exact population-block smoother is limited to blocks of 5000 states");
        if(p.smoother=="block-ilu") {
          block_ilu.push_back(std::make_unique<PopulationBlockILU>(A,first,last,
                                      p.block_kernel=="structured" && !space.reflected));
          continue;
        }
        std::vector<num::idx> row_ptr{0}, columns;
        std::vector<double> values;
        for (Index row=first;row<last;++row) {
          for (Index j=A.ptr[row];j<A.ptr[row+1];++j)
            if (A.column[j]>=first && A.column[j]<last) {
              columns.push_back(A.column[j]-first);
              values.push_back(A.value[j]);
            }
          row_ptr.push_back(values.size());
        }
        num::spmat block(last-first,last-first,
          num::array<double>(values.begin(),values.end()),
          num::array<num::idx>(columns.begin(),columns.end()),
          num::array<num::idx>(row_ptr.begin(),row_ptr.end()));
        block_work.emplace_back(last-first,0.);
        block_solution.emplace_back(last-first,0.);
        block_factors.push_back(std::make_unique<num::auto_linear_solver>(block));
      }
      if(p.smoother=="block-ilu")
        block_ilu_work.resize(block_offsets[p.cap+1]-block_offsets[p.cap]);
    }
    if ((p.smoother=="cascade" || p.smoother=="block-cascade") && V>1) {
      const auto start=Clock::now();
      if (p.smoother=="block-cascade") block_cascade.build(A,block_offsets,p.cascade_depth,p.cascade_rcond);
      else cascade.build(A,p.cascade_depth,p.cascade_row_cap);
      cascade_setup=seconds(start);
    }
    if (p.smoother=="chain" && V>1) build_chains(neighbors);
    if (p.smoother=="overlap" && V>1) build_overlap(neighbors);
    if (!hierarchy) return;
    if (p.cells==1) {
      if(previous) {
        pivot=std::move(previous->pivot); lower=std::move(previous->lower);
        upper=std::move(previous->upper);
      }
      pivot.resize(space.size); lower.resize(space.size,0.); upper.resize(space.size,0.);
      // The last old upper entry gains the coupling to the new population.
      // All preceding Thomas coefficients remain valid.
      for (Index i=old_size?old_size-1:0; i<space.size; ++i) {
        pivot[i]=A.value[A.diagonal[i]];
        for (Index j=A.ptr[i]; j<A.ptr[i+1]; ++j) {
          if (A.column[j]+1==i) lower[i]=A.value[j];
          if (A.column[j]==i+1) upper[i]=A.value[j];
        }
        if (i>0) {lower[i]/=pivot[i-1]; pivot[i]-=lower[i]*upper[i-1];}
      }
      return;
    }
    if (p.cells%2) throw std::invalid_argument("cells must be a power of two");
    auto cp=p;
    const int ratio=p.hierarchy=="direct"?p.cells:2;
    cp.cells/=ratio; cp.diffusion*=ratio;
    coarse=std::make_unique<Level>(cp,true,previous?previous->coarse.get():nullptr);
    if(previous) {
      parent=std::move(previous->parent); weight=std::move(previous->weight);
      reused_transfers=old_size;
    }
    parent.resize(space.size); weight.resize(space.size);
    coarse_rhs=Vec(coarse->space.size,0.); coarse_solution=coarse_rhs;
    if (p.cycle_type=="k" && coarse->coarse)
      krylov_v1=krylov_w1=krylov_v2=krylov_w2=krylov_r=coarse_rhs;
    std::vector<int> voxel_parent(V);
    for (int v=0; v<V; ++v) {
      int fine=v, stride=1, cv=0;
      for (int d=0; d<p.dimension; ++d) {
        cv+=((fine%p.cells)/ratio)*stride;
        fine/=p.cells; stride*=cp.cells;
      }
      voxel_parent[v]=cv;
    }
    std::vector<double> log_factorial(p.cap+1), sums(coarse->space.size,0.);
    for (int k=0;k<=p.cap;++k) log_factorial[k]=std::lgamma(k+1.);
    std::vector<int> cn(coarse->space.voxels);
    const double log_children=p.dimension*std::log(static_cast<double>(ratio));
    space.each([&](Index i,const std::vector<int>&n,int total) {
      std::fill(cn.begin(),cn.end(),0);
      double log_weight=-total*log_children;
      for (int v=0;v<V;++v) {cn[voxel_parent[v]]+=n[v];log_weight-=log_factorial[n[v]];}
      for (int k:cn) log_weight+=log_factorial[k];
      parent[i]=coarse->space.rank(cn,total);
      log_weight+=std::log(double(space.multiplicity(i))/coarse->space.multiplicity(parent[i]));
      weight[i]=std::exp(log_weight); sums[parent[i]]+=weight[i];
    }, old_cap+1);
    for (Index i=old_size;i<space.size;++i) weight[i]/=sums[parent[i]];
  }
  void build_chains(const std::vector<std::vector<int>> &neighbors) {
    const int V=space.voxels;
    // Edge (a,b), a<b, has id given by its position in this sorted list.
    std::vector<std::uint64_t> edges;
    for (int b=0;b<V;++b) for (int a:neighbors[b]) if (a<b) edges.push_back(std::uint64_t(a)*V+b);
    std::sort(edges.begin(),edges.end());
    struct Chain{Index edge, first, length;};
    std::vector<Chain> chains;
    std::vector<Index> members;
    std::vector<char> covered(space.size,0);
    std::vector<int> work(V);
    space.each([&](Index i,const std::vector<int> &n,int total) {
      for (int b=0;b<V;++b) {
        if (n[b]==0) continue;
        for (int a:neighbors[b]) {
          if (a>b || n[a]!=0) continue;     // chain start: n_a=0, n_b=m>=1
          const auto edge=static_cast<Index>(std::lower_bound(edges.begin(),edges.end(),
              std::uint64_t(a)*V+b)-edges.begin());
          const int m=n[b];
          work=n;
          chains.push_back({edge,static_cast<Index>(members.size()),static_cast<Index>(m+1)});
          for (int j=0;j<=m;++j) {
            work[a]=j; work[b]=m-j;
            const Index s=j==0?i:space.rank(work,total);
            members.push_back(s); covered[s]=1;
          }
        }
      }
    });
    std::stable_sort(chains.begin(),chains.end(),[](const Chain &x,const Chain &y){return x.edge<y.edge;});
    auto entry=[&](Index row,Index column) {
      const auto first=A.column.begin()+A.ptr[row], last=A.column.begin()+A.ptr[row+1];
      const auto it=std::lower_bound(first,last,column);
      if (it==last || *it!=column) throw std::logic_error("chain coupling missing from A");
      return A.value[it-A.column.begin()];
    };
    chain_ptr.assign(1,0);
    Index longest=0;
    for (const auto &c:chains) {
      double previous_ratio=0.;
      for (Index j=0;j<c.length;++j) {
        const Index s=members[c.first+j];
        const double lower=j>0?entry(s,members[c.first+j-1]):0.;
        const double upper=j+1<c.length?entry(s,members[c.first+j+1]):0.;
        const double pivot=A.value[A.diagonal[s]]-lower*previous_ratio;
        chain_state.push_back(s); chain_lower.push_back(lower);
        chain_inv_pivot.push_back(1./pivot); chain_upper_ratio.push_back(upper/pivot);
        previous_ratio=upper/pivot;
      }
      chain_ptr.push_back(static_cast<Index>(chain_state.size()));
      longest=std::max(longest,c.length);
    }
    for (Index i=0;i<space.size;++i) if (!covered[i]) chain_uncovered.push_back(i);
    chain_work=Vec(longest,0.);
  }
  void build_overlap(const std::vector<std::vector<int>> &neighbors) {
    const int V=space.voxels;
    std::vector<std::vector<int>> patches(V);
    for (int v=0;v<V;++v) {
      patches[v]=neighbors[v];
      patches[v].push_back(v);
      std::sort(patches[v].begin(),patches[v].end());
    }
    overlap_ptr.push_back(0);
    overlap_rowptr.push_back(0);
    std::vector<char> covered(space.size,0);
    std::vector<int> work(V);
    std::vector<Index> locate(space.size,std::numeric_limits<Index>::max());
    Index longest=0;
    space.each([&](Index, const std::vector<int> &n,int total) {
      for (const auto &patch:patches) {
        const int width=static_cast<int>(patch.size());
        const int m=n[patch[0]];
        if (m==0) continue;
        bool anchor=true;
        for (int j=1;j<width;++j) if (n[patch[j]]!=0) {anchor=false;break;}
        if (!anchor) continue;
        const auto length=space.choose[m+width-1][width-1];
        work=n;
        const Index first=static_cast<Index>(overlap_state.size());
        std::function<void(int,int)> enumerate=[&](int offset,int remaining) {
          if (offset==width-1) {
            work[patch[offset]]=remaining;
            const Index s=space.rank(work,total);
            overlap_state.push_back(s); covered[s]=1;
            return;
          }
          for (int count=0;count<=remaining;++count) {
            work[patch[offset]]=count;
            enumerate(offset+1,remaining-count);
          }
        };
        enumerate(0,m);
        if (overlap_state.size()-first!=length) throw std::logic_error("overlap block size mismatch");
        for (Index i=0;i<length;++i) locate[overlap_state[first+i]]=i;
        std::vector<Index> rowptr{0}, column, diagonal;
        std::vector<double> factor;
        std::vector<std::pair<Index,double>> entries;
        for (Index i=0;i<length;++i) {
          const Index row=overlap_state[first+i];
          entries.clear();
          for (Index j=A.ptr[row];j<A.ptr[row+1];++j) {
            const Index local=locate[A.column[j]];
            if (local!=std::numeric_limits<Index>::max())
              entries.emplace_back(local,A.value[j]);
          }
          std::sort(entries.begin(),entries.end());
          bool found_diagonal=false;
          for (auto [local,value]:entries) {
            if (local==i) {diagonal.push_back(static_cast<Index>(factor.size()));found_diagonal=true;}
            column.push_back(local);factor.push_back(value);
          }
          if (!found_diagonal) throw std::logic_error("overlap diagonal missing");
          rowptr.push_back(static_cast<Index>(factor.size()));
        }
        for (Index i=0;i<length;++i) locate[overlap_state[first+i]]=std::numeric_limits<Index>::max();
        std::vector<Index> scratch_index(length);
        if (!num::ilu0_factor(factor.data(),rowptr.data(),column.data(),diagonal.data(),
                              scratch_index.data(),static_cast<Index>(length)))
          throw std::runtime_error("overlap sparse ILU(0) pivot failure");
        if (overlap_factor.size()+factor.size()>50000000)
          throw std::runtime_error("overlap sparse factors exceed 50-million-entry safety limit");
        const Index factor_first=static_cast<Index>(overlap_factor.size());
        overlap_factor.insert(overlap_factor.end(),factor.begin(),factor.end());
        overlap_column.insert(overlap_column.end(),column.begin(),column.end());
        for (Index i=0;i<length;++i) {
          overlap_diag.push_back(factor_first+diagonal[i]);
          overlap_rowptr.push_back(factor_first+rowptr[i+1]);
        }
        overlap_ptr.push_back(static_cast<Index>(overlap_state.size()));
        longest=std::max(longest,static_cast<Index>(length));
      }
    });
    for (Index i=0;i<space.size;++i) if (!covered[i]) overlap_uncovered.push_back(i);
    overlap_work=Vec(longest,0.);
  }
  void overlap_sweep(const Vec &b,Vec &x,bool reverse) const {
    auto point=[&](Index row) {
      double r=b[row];
      for (Index j=A.ptr[row];j<A.ptr[row+1];++j) r-=A.value[j]*x[A.column[j]];
      x[row]+=r/A.value[A.diagonal[row]];
    };
    if (!reverse) for (Index row:overlap_uncovered) point(row);
    const Index count=static_cast<Index>(overlap_ptr.size()-1);
    for (Index visit=0;visit<count;++visit) {
      const Index c=reverse?count-1-visit:visit;
      const Index first=overlap_ptr[c], length=overlap_ptr[c+1]-first;
      for (Index i=0;i<length;++i) {
        const Index row=overlap_state[first+i];
        double r=b[row];
        for (Index j=A.ptr[row];j<A.ptr[row+1];++j) r-=A.value[j]*x[A.column[j]];
        overlap_work[i]=r;
      }
      for (Index i=0;i<length;++i) {
        double r=overlap_work[i];
        for (Index j=overlap_rowptr[first+i];j<overlap_diag[first+i];++j)
          r-=overlap_factor[j]*overlap_work[overlap_column[j]];
        overlap_work[i]=r;
      }
      for (Index i=length;i-->0;) {
        double r=overlap_work[i];
        for (Index j=overlap_diag[first+i]+1;j<overlap_rowptr[first+i+1];++j)
          r-=overlap_factor[j]*overlap_work[overlap_column[j]];
        overlap_work[i]=r/overlap_factor[overlap_diag[first+i]];
      }
      for (Index i=0;i<length;++i) x[overlap_state[first+i]]+=overlap_work[i];
    }
    if (reverse) for (auto it=overlap_uncovered.rbegin();it!=overlap_uncovered.rend();++it) point(*it);
  }
  void chain_sweep(const Vec &b,Vec &x,bool reverse) const {
    auto point=[&](Index row) {
      double sum=b[row];
      for (Index j=A.ptr[row];j<A.ptr[row+1];++j) if (A.column[j]!=row) sum-=A.value[j]*x[A.column[j]];
      x[row]=sum/A.value[A.diagonal[row]];
    };
    if (!reverse) for (Index row:chain_uncovered) point(row);
    const Index count=static_cast<Index>(chain_ptr.size()-1);
    for (Index visit=0;visit<count;++visit) {
      const Index c=reverse?count-1-visit:visit;
      const Index first=chain_ptr[c], length=chain_ptr[c+1]-first;
      // Forward elimination on the chain residual, then back substitution.
      for (Index j=0;j<length;++j) {
        const Index row=chain_state[first+j];
        double r=b[row];
        for (Index k=A.ptr[row];k<A.ptr[row+1];++k) r-=A.value[k]*x[A.column[k]];
        const double previous=j>0?chain_work[j-1]:0.;
        chain_work[j]=(r-chain_lower[first+j]*previous)*chain_inv_pivot[first+j];
      }
      for (Index j=length-1;j-->0;)
        chain_work[j]-=chain_upper_ratio[first+j]*chain_work[j+1];
      for (Index j=0;j<length;++j) x[chain_state[first+j]]+=chain_work[j];
    }
    if (reverse) for (auto it=chain_uncovered.rbegin();it!=chain_uncovered.rend();++it) point(*it);
  }
  void smooth(const Vec &b,Vec &x,bool reverse) const {
    ProfileTimer timer(p.profile,smooth_seconds);
    if (p.smoother=="cascade" && p.cells>1) {
      // Saye's reverse pre-smoothing / forward post-smoothing ordering.
      // Scalar diagonal factors equal their own transposes.
      for (int sweep=0;sweep<p.sweeps;++sweep) cascade.apply(A,b,x,scratch,!reverse && p.cascade_order=="reverse-pre");
      return;
    }
    if (p.smoother=="block-cascade" && p.cells>1) {
      for (int sweep=0;sweep<p.sweeps;++sweep)
        block_cascade.apply(A,b,x,scratch,!reverse && p.cascade_order=="reverse-pre");
      return;
    }
    if (p.smoother=="chain" && p.cells>1) {
      for (int sweep=0;sweep<p.sweeps;++sweep) chain_sweep(b,x,reverse);
      return;
    }
    if (p.smoother=="overlap" && p.cells>1) {
      for (int sweep=0;sweep<p.sweeps;++sweep) overlap_sweep(b,x,reverse);
      return;
    }
    if (p.cells==1 || (p.smoother!="block" && p.smoother!="block-ilu")) {
      A.smooth(b,x,p.sweeps,reverse,p.smoother=="jacobi",scratch);
      return;
    }
    for (int sweep=0;sweep<p.sweeps;++sweep)
      for (int visit=0;visit<=p.cap;++visit) {
        const int k=reverse?p.cap-visit:visit;
        const Index first=block_offsets[k],last=block_offsets[k+1];
        double *rhs=p.smoother=="block"?block_work[k].data():block_ilu_work.data();
        if(p.smoother=="block-ilu" && !block_ilu[k]->pivot.empty()) {
          block_ilu[k]->smooth(A,b,x,first,rhs);
          continue;
        }
        for (Index row=first;row<last;++row) {
          double sum=b[row];
          for (Index j=A.ptr[row];j<A.ptr[row+1];++j)
            sum-=A.value[j]*x[A.column[j]];
          rhs[row-first]=sum;
        }
        // ILU triangular substitution supports aliasing; every rhs entry is
        // overwritten on the next visit. Exact solves use a separate buffer.
        if (p.smoother=="block") {
          num::solve(*block_factors[k],block_work[k],block_solution[k]);
          for (Index row=first;row<last;++row)x[row]+=block_solution[k][row-first];
        } else {
          block_ilu[k]->apply(rhs);
          for (Index row=first;row<last;++row)x[row]+=rhs[row-first];
        }
      }
  }
  void cycle(const Vec &b, Vec &x) const {
    if(p.profile)++cycle_calls;
    if (!coarse) {
      std::copy(b.begin(),b.end(),x.begin());
      for (Index i=1;i<space.size;++i) x[i]-=lower[i]*x[i-1];
      x.back()/=pivot.back();
      for (Index i=space.size-1;i>0;--i) x[i-1]=(x[i-1]-upper[i-1]*x[i])/pivot[i-1];
      return;
    }
    smooth(b,x,false);
    {
      ProfileTimer timer(p.profile,matvec_seconds);
      A.apply(x,residual);
    }
    {
      ProfileTimer timer(p.profile,transfer_seconds);
      std::fill(coarse_rhs.begin(),coarse_rhs.end(),0.);
      for (Index i=0;i<space.size;++i) coarse_rhs[parent[i]]+=b[i]-residual[i];
    }
    std::fill(coarse_solution.begin(),coarse_solution.end(),0.);
    if (p.cycle_type=="k" && coarse->coarse) krylov_coarse_solve();
    // An exact bottom solve overwrites its input guess. Repeating it on a
    // W-cycle visit produces the same result without improving the correction.
    else for(int visit=0;visit<(coarse->coarse?p.cycles:1);++visit)
      coarse->cycle(coarse_rhs,coarse_solution);
    {
      ProfileTimer timer(p.profile,transfer_seconds);
      for (Index i=0;i<space.size;++i) x[i]+=weight[i]*coarse_solution[parent[i]];
    }
    smooth(b,x,true);
  }
  // K-cycle coarse solve (Notay--Vassilevski, nonsymmetric GCR form): at most
  // two minimal-residual iterations on A_c e = g, each preconditioned by one
  // recursive coarse cycle. The second iteration is skipped when the first
  // reduces the Euclidean residual by the factor 0.25. Below an exact bottom
  // solve no Krylov step is needed. The resulting cycle is nonlinear, so it is
  // used only as a standalone iteration, not as a fixed GMRES preconditioner.
  void krylov_coarse_solve() const {
    const Level &c=*coarse;
    const Index n=static_cast<Index>(coarse_rhs.size());
    auto dot=[n](const Vec &a,const Vec &b){double s=0.;for(Index i=0;i<n;++i)s+=a[i]*b[i];return s;};
    const double gg=dot(coarse_rhs,coarse_rhs);
    if (!(gg>0.)) return;
    c.apply(coarse_rhs,krylov_v1); c.A.apply(krylov_v1,krylov_w1);
    const double ww1=dot(krylov_w1,krylov_w1);
    if (!(ww1>0.)) return;
    const double a1=dot(krylov_w1,coarse_rhs)/ww1;
    for (Index i=0;i<n;++i) {
      coarse_solution[i]=a1*krylov_v1[i];
      krylov_r[i]=coarse_rhs[i]-a1*krylov_w1[i];
    }
    if (dot(krylov_r,krylov_r)<=0.0625*gg) return;
    c.apply(krylov_r,krylov_v2); c.A.apply(krylov_v2,krylov_w2);
    const double beta=dot(krylov_w2,krylov_w1)/ww1;
    for (Index i=0;i<n;++i) {krylov_v2[i]-=beta*krylov_v1[i];krylov_w2[i]-=beta*krylov_w1[i];}
    const double ww2=dot(krylov_w2,krylov_w2);
    if (!(ww2>0.)) return;
    const double a2=dot(krylov_w2,krylov_r)/ww2;
    for (Index i=0;i<n;++i) coarse_solution[i]+=a2*krylov_v2[i];
  }
  using domain_type=Vec;
  using codomain_type=Vec;
  num::idx rows() const{return space.size;}
  num::idx cols() const{return rows();}
  void apply(const Vec &b,Vec &x) const{std::fill(x.begin(),x.end(),0.);cycle(b,x);}
  double residual_error(const Vec &b,const Vec &x,Vec &r) const {
    ProfileTimer timer(p.profile,residual_seconds);
    return A.residual_norm(b,x,r)/p.shift;
  }
  double cascade_setup_seconds() const {return cascade_setup+(coarse?coarse->cascade_setup_seconds():0.);}
  void report_profile() const {
    if(!p.profile)return;
    std::uint64_t block_bytes=0;
    for(const auto &block:block_ilu)block_bytes+=block->bytes();
    std::cerr<<std::setprecision(9)<<"PROFILE voxels="<<space.voxels<<" states="<<rows()
      <<" cycles="<<cycle_calls<<" smooth_s="<<smooth_seconds
      <<" matvec_s="<<matvec_seconds<<" transfer_s="<<transfer_seconds
      <<" convergence_residual_s="<<residual_seconds<<" matrix_nnz="<<A.value.size()
      <<" block_factor_bytes="<<block_bytes<<'\n';
    if(coarse)coarse->report_profile();
  }
  std::uint64_t reused_setup() const {
    return reused_rows+reused_factors+reused_transfers+(coarse?coarse->reused_setup():0);
  }
  std::uint64_t block_workspace_bytes() const {
    std::uint64_t result=8*block_ilu_work.size();
    for(const auto &v:block_work)result+=8*v.size();
    for(const auto &v:block_solution)result+=8*v.size();
    return result;
  }
  std::uint64_t bytes() const {
    std::uint64_t factor_bytes=0;
    for(const auto &block:block_ilu)factor_bytes+=block->bytes();
    return 4*(space.folded.size()+space.offsets.size())+space.orbit_size.size()+
        factor_bytes+block_workspace_bytes()+cascade.bytes()+block_cascade.bytes()+A.bytes()+12*parent.size()+8*(residual.size()+correction.size()+scratch.size()+
        coarse_rhs.size()+coarse_solution.size()+leakage.size()+lower.size()+upper.size()+pivot.size()+
        krylov_v1.size()+krylov_w1.size()+krylov_v2.size()+krylov_w2.size()+krylov_r.size()+
        chain_lower.size()+chain_inv_pivot.size()+chain_upper_ratio.size()+chain_work.size()+
        overlap_factor.size()+overlap_work.size())+
        4*(chain_ptr.size()+chain_state.size()+chain_uncovered.size()+overlap_ptr.size()+
           overlap_state.size()+overlap_rowptr.size()+overlap_column.size()+
           overlap_diag.size()+overlap_uncovered.size())+
        2*population.size()+(coarse?coarse->bytes():0);
  }
  void check() const {
    if(p.smoother=="block-ilu" && p.block_kernel=="structured" &&
       p.cells>1 && !space.reflected && rows()<=5000) {
      Vec b(rows(),0.),original(rows(),0.);
      for(Index i=0;i<rows();++i) {
        b[i]=std::cos(.13*i); original[i]=std::sin(.37*i);
      }
      double maximum=0.;
      for(int k=0;k<=p.cap;++k) {
        const Index first=block_offsets[k],last=block_offsets[k+1];
        PopulationBlockILU reference(A,first,last,false);
        std::vector<double> work(last-first),fused_work(last-first);
        for(Index row=first;row<last;++row) {
          const Index i=row-first;
          const double expected=reference.factor.value[reference.factor.diagonal[i]];
          if(std::abs(1.-expected*block_ilu[k]->pivot[i])>1e-13)
            throw std::runtime_error("structured ILU pivot differs from generic factorization");
          double sum=b[row];
          for(Index j=A.ptr[row];j<A.ptr[row+1];++j)sum-=A.value[j]*original[A.column[j]];
          work[i]=sum;
        }
        reference.apply(work.data());
        Vec actual=original;
        block_ilu[k]->smooth(A,b,actual,first,fused_work.data());
        for(Index row=first;row<last;++row) {
          const double expected=original[row]+work[row-first];
          maximum=std::max(maximum,std::abs(actual[row]-expected)/std::max(1.,std::abs(expected)));
        }
      }
      if(maximum>1e-12)throw std::runtime_error("fused block update differs from reference ILU");
      std::cerr<<"check structured_ilu relative_update_error="<<maximum<<'\n';
    }
    Vec column_sum(rows(),0.);
    for(Index i=0;i<rows();++i)for(Index j=A.ptr[i];j<A.ptr[i+1];++j){
      column_sum[A.column[j]]+=A.value[j];
      if(A.column[j]!=i && A.value[j]>0.)throw std::runtime_error("positive off-diagonal in A");
    }
    for(Index i=0;i<rows();++i)
      if(std::abs(column_sum[i]-p.shift-leakage[i])>1e-10*A.value[A.diagonal[i]])
        throw std::runtime_error("generator mass balance check failed");
    if (!coarse) return;
    Vec q(coarse->rows(),0.), y(rows(),0.), Ay(rows(),0.), coarse_image(coarse->rows(),0.),
        restricted(coarse->rows(),0.);
    // A deterministic signed vector exercises every coarse column, including
    // population zero and cap. This is an independent fine/coarse assembly check.
    for (Index j=0;j<q.size();++j) q[j]=std::sin(.7*j)+.25*std::cos(.31*j);
    for (Index i=0;i<rows();++i) {y[i]=weight[i]*q[parent[i]];restricted[parent[i]]+=y[i];}
    double rp=0.;for(Index j=0;j<q.size();++j) rp=std::max(rp,std::abs(q[j]-restricted[j]));
    A.apply(y,Ay);coarse->A.apply(q,coarse_image);
    std::fill(restricted.begin(),restricted.end(),0.);
    for(Index i=0;i<rows();++i) restricted[parent[i]]+=Ay[i];
    double err=0.,scale=0.;
    for(Index j=0;j<q.size();++j){err=std::max(err,std::abs(restricted[j]-coarse_image[j]));scale=std::max(scale,std::abs(coarse_image[j]));}
    std::cerr<<"check cells="<<p.cells<<" RP="<<rp<<" RAP="<<err/std::max(1.,scale)<<'\n';
    if(rp>1e-10||err>1e-10*std::max(1.,scale))throw std::runtime_error("coarse operator check failed");
    coarse->check();
  }
};


// Export the stored transfers in the solver's state ordering. One row records
// R[coarse_state,fine_state]=1 and P[fine_state,coarse_state]=weight.
void export_transfers(const Level &fine,const std::string &prefix) {
  std::ofstream manifest(prefix+"_levels.csv");
  if(!manifest)throw std::runtime_error("cannot write transfer manifest");
  manifest<<"level,fine_voxels,coarse_voxels,cap,fine_states,coarse_states,nnz,reflection\n";
  int index=0;
  for(const Level *level=&fine;level->coarse;level=level->coarse.get(),++index) {
    std::ofstream out(prefix+"_level"+std::to_string(index)+".csv");
    if(!out)throw std::runtime_error("cannot write transfer entries");
    out<<std::setprecision(17)<<"fine_state,coarse_state,population,weight\n";
    for(Index i=0;i<level->rows();++i)
      out<<i<<','<<level->parent[i]<<','<<level->population[i]<<','<<level->weight[i]<<'\n';
    if(!out)throw std::runtime_error("failed while writing transfer entries");
    manifest<<index<<','<<level->space.voxels<<','<<level->coarse->space.voxels<<','
      <<level->p.cap<<','<<level->rows()<<','<<level->coarse->rows()<<','
      <<level->parent.size()<<','<<level->space.reflected<<'\n';
  }
  if(!manifest)throw std::runtime_error("failed while writing transfer manifest");
}

struct ILU {
  using domain_type=Vec; using codomain_type=Vec;
  const Matrix &A; std::vector<double> values;
  explicit ILU(const Matrix &a):A(a),values(a.value){
    std::vector<Index> scratch(A.rows());
    if(!num::ilu0_factor(values.data(),A.ptr.data(),A.column.data(),A.diagonal.data(),scratch.data(),static_cast<Index>(A.rows())))
      throw std::runtime_error("ILU zero pivot");
  }
  num::idx rows()const{return A.rows();}num::idx cols()const{return rows();}
  void apply(const Vec&b,Vec&x)const{
    num::csr_lu_solve(x.data(),values.data(),A.ptr.data(),A.column.data(),A.diagonal.data(),b.data(),static_cast<Index>(rows()));
  }
};

// Bound SuiteSparse's allocations inside this benchmark process. A failed
// allocation is reported by KLU, rather than allowing a fill-heavy comparison
// to exhaust the user's machine. The fine CSR and process workspace are
// outside this 2 GiB factorization budget and are included in measured RSS.
namespace factor_budget {
constexpr std::size_t limit=std::size_t{2}<<30;
std::size_t used=0,peak=0;
bool refused=false;
struct alignas(std::max_align_t) Header{std::size_t bytes;};
void *allocate(std::size_t n){
  if(n>limit-used){refused=true;return nullptr;}
  auto *h=static_cast<Header*>(std::malloc(n+sizeof(Header)));
  if(!h)return nullptr;
  h->bytes=n;used+=n;peak=std::max(peak,used);return h+1;
}
void release(void *p){if(p){auto *h=static_cast<Header*>(p)-1;used-=h->bytes;std::free(h);}}
void *zero_allocate(std::size_t n,std::size_t size){
  if(size && n>limit/size){refused=true;return nullptr;}
  void *p=allocate(n*size);if(p)std::memset(p,0,n*size);return p;
}
void *resize(void *p,std::size_t n){
  if(!p)return allocate(n);
  if(!n){release(p);return nullptr;}
  auto *h=static_cast<Header*>(p)-1;
  const auto old=h->bytes;
  if(n>limit-(used-old)){refused=true;return nullptr;}
  auto *next=static_cast<Header*>(std::realloc(h,n+sizeof(Header)));
  if(!next)return nullptr;
  next->bytes=n;used=used-old+n;peak=std::max(peak,used);return next+1;
}
struct Guard{
  Guard(){
    SuiteSparse_config_malloc_func_set(allocate);SuiteSparse_config_calloc_func_set(zero_allocate);
    SuiteSparse_config_realloc_func_set(resize);SuiteSparse_config_free_func_set(release);
  }
  ~Guard(){
    SuiteSparse_config_malloc_func_set(std::malloc);SuiteSparse_config_calloc_func_set(std::calloc);
    SuiteSparse_config_realloc_func_set(std::realloc);SuiteSparse_config_free_func_set(std::free);
  }
};
}
struct Direct{
  factor_budget::Guard guard;
  klu_common common{};klu_symbolic *symbolic=nullptr;klu_numeric *numeric=nullptr;
  klu_l_common long_common{};klu_l_symbolic *long_symbolic=nullptr;klu_l_numeric *long_numeric=nullptr;
  explicit Direct(const Matrix&A){
    klu_defaults(&common);
    std::vector<std::int32_t> ptr(A.rows()+1,0), row(A.value.size());
    std::vector<double> values(A.value.size());
    for(Index c:A.column)++ptr[c+1];
    std::partial_sum(ptr.begin(),ptr.end(),ptr.begin());
    auto next=ptr;
    for(Index i=0;i<A.rows();++i)for(Index j=A.ptr[i];j<A.ptr[i+1];++j){
      const auto pos=next[A.column[j]]++;row[pos]=i;values[pos]=A.value[j];
    }
    symbolic=klu_analyze(A.rows(),ptr.data(),row.data(),&common);
    if(!symbolic)throw std::runtime_error("KLU symbolic analysis failed");
    if(symbolic->lnz<0||symbolic->unz<0){
      // Input indices fit in 32 bits, but the predicted factors need not.
      klu_free_symbolic(&symbolic,&common);klu_l_defaults(&long_common);
      std::vector<std::int64_t> lp(ptr.begin(),ptr.end()),lr(row.begin(),row.end());
      long_symbolic=klu_l_analyze(A.rows(),lp.data(),lr.data(),&long_common);
      if(!long_symbolic)throw std::runtime_error("64-bit KLU symbolic analysis failed");
      std::cerr<<"KLU index_bits=64 estimated_lnz="<<long_symbolic->lnz<<" estimated_unz="<<long_symbolic->unz<<'\n';
      long_numeric=klu_l_factor(lp.data(),lr.data(),values.data(),long_symbolic,&long_common);
      std::cerr<<"KLU factor_peak_bytes="<<factor_budget::peak<<" allocation_budget_exceeded="<<factor_budget::refused<<'\n';
      if(!long_numeric){klu_l_free_symbolic(&long_symbolic,&long_common);throw std::runtime_error("64-bit KLU factorization failed within the 2 GiB allocation budget");}
      return;
    }
    std::cerr<<"KLU estimated_lnz="<<symbolic->lnz<<" estimated_unz="<<symbolic->unz<<'\n';
    numeric=klu_factor(ptr.data(),row.data(),values.data(),symbolic,&common);
    std::cerr<<"KLU factor_peak_bytes="<<factor_budget::peak<<" allocation_budget_exceeded="<<factor_budget::refused<<'\n';
    if(!numeric){
      klu_free_symbolic(&symbolic,&common);
      throw std::runtime_error(factor_budget::refused?"KLU exceeded the 2 GiB factorization allocation budget":"KLU numeric factorization failed");
    }
    std::cerr<<"KLU actual_lnz="<<numeric->lnz<<" actual_unz="<<numeric->unz<<'\n';
  }
  ~Direct(){
    klu_free_numeric(&numeric,&common);klu_free_symbolic(&symbolic,&common);
    klu_l_free_numeric(&long_numeric,&long_common);klu_l_free_symbolic(&long_symbolic,&long_common);
  }
  void apply(const Vec&b,Vec&x){
    x=b;
    const bool success=long_numeric?klu_l_solve(long_symbolic,long_numeric,b.size(),1,x.data(),&long_common):klu_solve(symbolic,numeric,b.size(),1,x.data(),&common);
    if(!success)throw std::runtime_error("KLU solve failed");
  }
};

int run(Parameters p) {
  const bool use_mg=p.method=="mg"||p.method=="mg-gmres";
  auto start=Clock::now();
  Level level(p,use_mg);
  const double assembly=seconds(start);
  if(p.check) level.check();
  if(!p.transfer_output.empty())export_transfers(level,p.transfer_output);
  if(p.export_only)return 0;
  start=Clock::now();
  std::unique_ptr<ILU> ilu;
  std::unique_ptr<Direct> sparse_direct;
  if(p.method=="ilu") ilu=std::make_unique<ILU>(level.A);
  if(p.method=="klu") sparse_direct=std::make_unique<Direct>(level.A);
  const double extra_setup=seconds(start);
  Vec probability(level.rows(),0.),rhs(level.rows(),0.),x(level.rows(),0.),residual(level.rows(),0.);
  std::vector<int> initial(level.space.voxels,0);
  for(int k=0;k<p.initial;++k)++initial[k%initial.size()];
  probability[level.space.rank(initial,p.initial)]=1.;
  auto save_state=[&](){
    if(p.state_output.empty())return;
    std::ofstream out(p.state_output);
    if(!out)throw std::runtime_error("cannot write final state vector");
    level.space.write_state(probability,out);
  };
  std::ofstream histogram;
  if(!p.output.empty()){histogram.open(p.output);histogram<<std::setprecision(15)<<"time,population,probability\n";}
  if(p.method=="uniformization"){
    if(level.rows()>5000)throw std::invalid_argument("uniformization is a small reference calculation");
    double lambda=0.;
    for(Index i=0;i<level.rows();++i)lambda=std::max(lambda,level.A.value[level.A.diagonal[i]]-p.shift);
    const double end_time=p.steps/p.shift;
    const int chunks=std::max(1,static_cast<int>(std::ceil(lambda*end_time/p.uniformization_mean)));
    const double dt=end_time/chunks,mu=lambda*dt;
    double tail_bound=0.;
    start=Clock::now();
    for(int chunk=0;chunk<chunks;++chunk){
      x=probability;double weight=std::exp(-mu);
      for(Index i=0;i<level.rows();++i)rhs[i]=weight*x[i];
      for(int term=1;lambda>0&&term<=80;++term){
        level.A.apply(x,residual);
        for(Index i=0;i<level.rows();++i)x[i]+=(p.shift*x[i]-residual[i])/lambda;
        weight*=mu/term;
        for(Index i=0;i<level.rows();++i)rhs[i]+=weight*x[i];
      }
      tail_bound+=weight*(mu/81.)/(1.-mu/82.);
      probability=rhs;
    }
    std::vector<double> marginal(p.cap+1,0.);
    for(Index i=0;i<level.rows();++i)marginal[level.population[i]]+=probability[i];
    if(histogram)for(int k=0;k<=p.cap;++k)histogram<<end_time<<','<<k<<','<<marginal[k]<<'\n';
    std::cout<<std::setprecision(15)<<"states,time,chunks,matvecs,poisson_tail_bound,mass,solve_s\n"
      <<level.rows()<<','<<end_time<<','<<chunks<<','<<(lambda>0?80*chunks:0)<<','<<tail_bound<<','
      <<std::accumulate(probability.begin(),probability.end(),0.)<<','<<seconds(start)<<'\n';
    save_state();
    return 0;
  }
  std::ofstream history;
  if(!p.history.empty()){history.open(p.history);history<<std::setprecision(15)<<"step,iteration,residual_l1_bound,elapsed_s\n";}
  std::unique_ptr<num::auto_linear_solver> direct;
  if(p.check&&level.rows()<=5000&&!sparse_direct){
    const auto &A=level.A;
    num::spmat matrix(A.rows(),A.cols(),num::array<double>(A.value.begin(),A.value.end()),
      num::array<num::idx>(A.column.begin(),A.column.end()),num::array<num::idx>(A.ptr.begin(),A.ptr.end()));
    direct=std::make_unique<num::auto_linear_solver>(matrix);
  }
  std::cout<<std::setprecision(12)<<"dimension,cells,cap,states,diffusion,shift,method,smoother,sweeps,cycles,step,iterations,residual_l1_bound,converged,setup_s,solve_s,mass,high_mass,min_probability,storage_bytes,peak_rss_bytes,cycle_type,cascade_depth,cascade_row_cap,cascade_setup_s,cascade_order,cascade_rcond,full_states\n";
  const int solves=p.steps+(p.integrator=="cn"?1:0);
  for(int step=1;step<=solves;++step){
    for(Index i=0;i<level.rows();++i)rhs[i]=p.shift*probability[i];
    // Rannacher start: two half backward-Euler steps, then Crank--Nicolson.
    // Both use the same matrix, with shift=2/dt. No refactorization is needed.
    bool initial_residual_ready=false;
    if(p.integrator=="cn"&&step>2){
      level.A.apply(probability,residual);
      for(Index i=0;i<level.rows();++i) {
        rhs[i]=2*p.shift*probability[i]-residual[i];
        residual[i]=rhs[i]-residual[i];
      }
      initial_residual_ready=true;
    }
    std::copy(probability.begin(),probability.end(),x.begin());
    int iterations=0;
    start=Clock::now();
    double error=initial_residual_ready?norm1(residual)/p.shift:level.residual_error(rhs,x,residual);
    while(iterations<p.max_iterations){
      bool stalled=false;
      if(history)history<<step<<','<<iterations<<','<<error<<','<<seconds(start)<<'\n';
      if(error<p.tolerance||!std::isfinite(error)||error>1e8)break;
      if(sparse_direct){sparse_direct->apply(residual,level.correction);for(Index i=0;i<level.rows();++i)x[i]+=level.correction[i];++iterations;}
      else if(p.method=="mg"){level.cycle(rhs,x);++iterations;}
      else if(p.method=="gs"){
        level.smooth(rhs,x,false);
        level.smooth(rhs,x,true);
        ++iterations;
      }else{
        // GMRES uses an absolute Euclidean residual tolerance. Enforce the same
        // independently recomputed l1 solution-error bound for every method.
        num::gmres_options options{.tolerance=p.tolerance*p.shift/std::sqrt(static_cast<double>(level.rows())),
          .max_iterations=static_cast<num::idx>(p.max_iterations-iterations),.restart=static_cast<num::idx>(p.restart)};
        num::solver_result result;
        if(ilu) result=num::gmres(level.A,*ilu,rhs,x,options);
        else if(use_mg) result=num::gmres(level.A,level,rhs,x,options);
        else result=num::gmres(level.A,rhs,x,options);
        iterations+=result.iterations;
        stalled=result.iterations==0;
      }
      // This residual is also the final convergence check. Reuse it after
      // breaking rather than applying the fine matrix a second time.
      error=level.residual_error(rhs,x,residual);
      if(stalled)break;
    }
    const double solve=seconds(start);
    if(direct){
      Vec reference(level.rows(),0.);num::solve(*direct,rhs,reference);
      for(Index i=0;i<level.rows();++i)reference[i]-=x[i];
      const double difference=norm1(reference);
      std::cerr<<"direct l1_error="<<difference<<" residual_bound="<<error<<'\n';
      if(difference>error+1e-10)throw std::runtime_error("direct reference disagrees with error bound");
    }
    double mass=0.,high=0.;std::vector<double> marginal(p.cap+1,0.);
    for(Index i=0;i<level.rows();++i){mass+=x[i];marginal[level.population[i]]+=x[i];if(level.population[i]>=21)high+=x[i];}
    rusage usage{};getrusage(RUSAGE_SELF,&usage);
    std::uint64_t rss=usage.ru_maxrss;
#ifndef __APPLE__
    rss*=1024;
#endif
    std::cout<<p.dimension<<','<<p.cells<<','<<p.cap<<','<<level.rows()<<','<<p.diffusion<<','<<p.shift<<','<<p.method<<','<<p.smoother<<','<<p.sweeps<<','<<p.cycles<<','<<step<<','<<iterations<<','<<error<<','<<(error<p.tolerance)<<','<<assembly+extra_setup<<','<<solve<<','<<mass<<','<<high<<','<<*std::min_element(x.begin(),x.end())<<','<<level.bytes()+(ilu?8*ilu->values.size():0)<<','<<rss<<','<<p.cycle_type<<','<<((p.smoother=="cascade" || p.smoother=="block-cascade")?p.cascade_depth:0)<<','<<p.cascade_row_cap<<','<<level.cascade_setup_seconds()<<','<<p.cascade_order<<','<<p.cascade_rcond<<','<<level.space.full_size<<std::endl;
    const double time=p.integrator=="cn"?(step==1?1.:2.*(step-1))/p.shift:step/p.shift;
    if(histogram)for(int k=0;k<=p.cap;++k)histogram<<time<<','<<k<<','<<marginal[k]<<'\n';
    if(!(error<p.tolerance))return 2;
    std::swap(probability,x);
  }
  save_state();
  level.report_profile();
  return 0;
}

// Adaptive whole-population FSP. Expansion is an accuracy decision, not a
// maximum state count. Backward Euler is used so each exact discrete step
// is positive and contractive in l1. The loss budget below concerns this
// discrete evolution; time-discretization error is a separate quantity.
int run_adaptive_fsp(Parameters p) {
  if(p.method!="mg"||p.integrator!="be")
    throw std::invalid_argument("adaptive FSP currently uses --method mg --integrator be");
  const double solve_tolerance=std::min(p.tolerance,p.fsp_tolerance/(100.*p.steps));
  const double dt=1./p.shift;
  if(!std::isfinite(dt*p.steps))throw std::invalid_argument("adaptive time horizon must be finite");
  std::unique_ptr<Level> level;
  Vec probability;
  double accumulated_error=0.,total_setup=0.,total_solve=0.;
  int expansions=0,attempt=0;
  std::ofstream histogram,history;
  if(!p.output.empty()) {
    histogram.open(p.output);
    if(!histogram)throw std::runtime_error("cannot write population output");
    histogram<<std::setprecision(17)<<"time,population,probability\n";
  }
  if(!p.history.empty()) {
    history.open(p.history);
    if(!history)throw std::runtime_error("cannot write adaptive history");
    history<<std::setprecision(17)<<"attempt,step,iteration,residual_l1_bound,elapsed_s\n";
  }
  std::cout<<std::setprecision(17)<<"attempt,step,time,cap,states,accepted,iterations,residual_l1_bound,missing_mass,algebraic_error_bound,loss_budget,setup_s,solve_s,total_setup_s,total_solve_s,peak_rss_bytes,full_states\n";
  for(int step=1;step<=p.steps;++step) {
    bool accepted=false;
    while(!accepted) {
      ++attempt;
      double setup=0.;
      if(!level||level->p.cap!=p.cap) {
        // Population ordering preserves old indices, blocks, and transfers.
        // Move retained storage into the enlarged hierarchy; rebuild only the
        // old boundary rows and append the new population blocks.
        auto start=Clock::now();
        if(p.reuse_fsp && level && Level::supports_population_reuse(p)) {
          auto previous=std::move(level);
          level=std::make_unique<Level>(p,true,previous.get());
          std::cerr<<"FSP reused rows="<<level->reused_rows
                   <<" block_factors="<<level->reused_factors
                   <<" transfer_entries="<<level->reused_transfers
                   <<" hierarchy_setup_items="<<level->reused_setup()<<'\n';
        } else {
          level.reset();
          level=std::make_unique<Level>(p,true);
        }
        if(p.check)level->check();
        Vec extended(level->rows(),0.);
        if(probability.size())std::copy(probability.begin(),probability.end(),extended.begin());
        else {
          std::vector<int> initial(level->space.voxels,0);
          for(int k=0;k<p.initial;++k)++initial[k%initial.size()];
          extended[level->space.rank(initial,p.initial)]=1.;
        }
        probability=std::move(extended);
        setup=seconds(start);total_setup+=setup;
      }
      Vec rhs(level->rows(),0.),trial=probability,residual(level->rows(),0.);
      for(Index i=0;i<level->rows();++i)rhs[i]=p.shift*probability[i];
      auto start=Clock::now();
      int iterations=0;
      double error=0.;
      while(true) {
        error=level->residual_error(rhs,trial,residual);
        if(history)history<<attempt<<','<<step<<','<<iterations<<','<<error<<','<<seconds(start)<<'\n';
        if(error<=solve_tolerance)break;
        if(iterations>=p.max_iterations||!std::isfinite(error))
          throw std::runtime_error("adaptive FSP linear solve failed; no step was accepted");
        level->cycle(rhs,trial);++iterations;
      }
      const double solve=seconds(start);total_solve+=solve;
      double mass=0.,negative_mass=0.;
      for(double value:trial) {
        mass+=value;
        if(value<0.)negative_mass-=value;
      }
      if(!std::isfinite(mass)||negative_mass>error+1e-13)
        throw std::runtime_error("adaptive FSP produced invalid probabilities; no step was accepted");
      // Contractivity bounds accumulated fine-vector solve error by the sum
      // of accepted local errors. Rejected trials never enter this ledger.
      const double next_error=accumulated_error+error;
      const double missing=std::max(0.,1.-mass);
      const double budget=p.fsp_tolerance*double(step)/p.steps;
      // One copy bounds the uncertainty in missing mass, and a second
      // bounds the returned vector's distance from the exact discrete FSP.
      accepted=missing+2*next_error<=budget;
      rusage usage{};getrusage(RUSAGE_SELF,&usage);
      std::uint64_t rss=usage.ru_maxrss;
#ifndef __APPLE__
      rss*=1024;
#endif
      std::cout<<attempt<<','<<step<<','<<step*dt<<','<<p.cap<<','<<level->rows()<<','<<accepted<<','<<iterations<<','<<error<<','<<missing<<','<<next_error<<','<<budget<<','<<setup<<','<<solve<<','<<total_setup<<','<<total_solve<<','<<rss<<','<<level->space.full_size<<std::endl;
      if(accepted) {
        probability=std::move(trial);accumulated_error=next_error;
        if(histogram) {
          std::vector<double> marginal(p.cap+1,0.);
          for(Index i=0;i<level->rows();++i)marginal[level->population[i]]+=probability[i];
          for(int k=0;k<=p.cap;++k)histogram<<step*dt<<','<<k<<','<<marginal[k]<<'\n';
        }
      } else {
        // Retry this same time interval from probability, never from trial.
        // Already accepted probability loss remains in the cumulative budget.
        const int new_cap=std::max(p.cap+4,static_cast<int>(std::ceil(1.25*(p.cap+1))));
        if(new_cap>10000)throw std::runtime_error("adaptive expansion exceeded the implementation's population resource guard");
        std::cerr<<"expand step="<<step<<" cap="<<p.cap<<" -> "<<new_cap<<" missing_mass="<<missing<<" budget="<<budget<<'\n';
        p.cap=new_cap;++expansions;
      }
    }
  }
  if(!p.state_output.empty()) {
    std::ofstream out(p.state_output);
    if(!out)throw std::runtime_error("cannot write final state vector");
    level->space.write_state(probability,out);
  }
  std::cerr<<"adaptive FSP complete: cap="<<p.cap<<" expansions="<<expansions
    <<"; loss budget excludes time-discretization error\n";
  level->report_profile();
  return 0;
}
} // namespace

int main(int argc,char **argv){
  try{
    Parameters p;
    bool cap_specified=false;
    for(int i=1;i<argc;++i){
      std::string arg=argv[i];
      if(arg=="--check"){p.check=true;continue;}
      if(arg=="--export-only"){p.export_only=true;continue;}
      if(arg=="--profile"){p.profile=true;continue;}
      if(arg=="--adaptive-fsp"){p.adaptive_fsp=true;continue;}
      if(i+1==argc)throw std::invalid_argument("missing value for "+arg);
      std::string value=argv[++i];
      if(arg=="--cells")p.cells=std::stoi(value);
      else if(arg=="--dimension")p.dimension=std::stoi(value);
      else if(arg=="--cap"){p.cap=std::stoi(value);cap_specified=true;}
      else if(arg=="--reuse-fsp")p.reuse_fsp=std::stoi(value)!=0;
      else if(arg=="--block-kernel")p.block_kernel=value;
      else if(arg=="--reflection")p.reflection=std::stoi(value)!=0;
      else if(arg=="--fsp-tolerance")p.fsp_tolerance=std::stod(value);
      else if(arg=="--diffusion")p.diffusion=std::stod(value);
      else if(arg=="--shift")p.shift=std::stod(value);
      else if(arg=="--steps")p.steps=std::stoi(value);
      else if(arg=="--initial")p.initial=std::stoi(value);
      else if(arg=="--sweeps")p.sweeps=std::stoi(value);
      else if(arg=="--cycles")p.cycles=std::stoi(value);
      else if(arg=="--cascade-rcond")p.cascade_rcond=std::stod(value);
      else if(arg=="--cascade-order")p.cascade_order=value;
      else if(arg=="--cascade-depth")p.cascade_depth=std::stoi(value);
      else if(arg=="--cascade-row-cap")p.cascade_row_cap=std::stoi(value);
      else if(arg=="--max-iterations")p.max_iterations=std::stoi(value);
      else if(arg=="--restart")p.restart=std::stoi(value);
      else if(arg=="--tolerance")p.tolerance=std::stod(value);
      else if(arg=="--uniformization-mean")p.uniformization_mean=std::stod(value);
      else if(arg=="--method")p.method=value;
      else if(arg=="--smoother")p.smoother=value;
      else if(arg=="--integrator")p.integrator=value;
      else if(arg=="--hierarchy")p.hierarchy=value;
      else if(arg=="--cycle-type")p.cycle_type=value;
      else if(arg=="--output")p.output=value;
      else if(arg=="--history")p.history=value;
      else if(arg=="--state-output")p.state_output=value;
      else if(arg=="--transfer-output")p.transfer_output=value;
      else if(arg=="--c1")p.c1=std::stod(value);
      else if(arg=="--c2")p.c2=std::stod(value);
      else if(arg=="--c3")p.c3=std::stod(value);
      else if(arg=="--c4")p.c4=std::stod(value);
      else if(arg=="--rates"){double scale=std::stod(value);p.c1*=scale;p.c2*=scale;p.c3*=scale;p.c4*=scale;}
      else if(arg=="--linear"){double scale=std::stod(value);p.c1=p.c2=0.;p.c3=2*scale;p.c4=scale;}
      else throw std::invalid_argument("unknown argument "+arg);
    }
    if(p.export_only&&p.transfer_output.empty())
      throw std::invalid_argument("--export-only requires --transfer-output");
    if(!p.transfer_output.empty()&&(p.adaptive_fsp||(p.method!="mg"&&p.method!="mg-gmres")))
      throw std::invalid_argument("transfer export requires a fixed multigrid hierarchy");
    if(p.adaptive_fsp&&!cap_specified)p.cap=std::max(2,p.initial);
    if(p.cells<1||(p.cells&(p.cells-1))||p.dimension<1||p.dimension>3||p.cap<0||p.cap>10000||p.initial<0||p.initial>p.cap||p.shift<=0||!std::isfinite(p.shift)||p.steps<1||p.sweeps<1||p.cycles<1||p.cycles>2||p.tolerance<=0||!std::isfinite(p.tolerance)||p.max_iterations<1||p.restart<1||p.diffusion<0||p.c1<0||p.c2<0||p.c3<0||p.c4<0||p.uniformization_mean<=0||p.uniformization_mean>8||p.fsp_tolerance<=0||p.fsp_tolerance>=1||!std::isfinite(p.fsp_tolerance))
      throw std::invalid_argument("invalid grid, population, shift or sweep count");
    if(p.method!="mg"&&p.method!="mg-gmres"&&p.method!="ilu"&&p.method!="gs"&&p.method!="gmres"&&p.method!="klu"&&p.method!="uniformization")
      throw std::invalid_argument("unknown method");
    if(p.smoother!="gs"&&p.smoother!="jacobi"&&p.smoother!="block"&&p.smoother!="block-ilu"&&p.smoother!="chain"&&p.smoother!="overlap"&&p.smoother!="cascade"&&p.smoother!="block-cascade")throw std::invalid_argument("unknown smoother");
    if(p.block_kernel!="structured" && p.block_kernel!="reference")
      throw std::invalid_argument("unknown population-block kernel");
    if(!(p.cascade_rcond>0. && p.cascade_rcond<1.) || !std::isfinite(p.cascade_rcond))
      throw std::invalid_argument("cascade SVD cutoff must lie in (0,1)");
    if(p.cascade_order.empty())
      p.cascade_order=p.smoother=="block-cascade"?"forward":"reverse-pre";
    if(p.cascade_order!="reverse-pre" && p.cascade_order!="forward")
      throw std::invalid_argument("cascade order must be reverse-pre or forward");
    if(p.smoother=="block-cascade" && (p.cascade_depth>3 || p.cascade_row_cap!=0))
      throw std::invalid_argument("block cascade uses depth 1..3 and no row truncation");
    if(p.cascade_depth<1||p.cascade_depth>4||p.cascade_row_cap<0)
      throw std::invalid_argument("invalid cascade depth or row cap");
    if(p.integrator!="be"&&p.integrator!="cn")throw std::invalid_argument("unknown integrator");
    if(p.hierarchy!="recursive"&&p.hierarchy!="direct")throw std::invalid_argument("unknown hierarchy");
    if(p.cycle_type!="standard"&&p.cycle_type!="k")throw std::invalid_argument("unknown cycle type");
    if(p.cycle_type=="k"&&p.method=="mg-gmres")
      throw std::invalid_argument("the K-cycle is nonlinear and cannot precondition standard GMRES");
    if(p.reflection) {
      if(p.initial%Level::voxel_count(p)!=0)
        throw std::invalid_argument("reflection requires an invariant initial configuration");
      if(!Level::supports_population_reuse(p))
        throw std::invalid_argument("reflection currently supports point and population-block smoothers");
    }
    return p.adaptive_fsp?run_adaptive_fsp(p):run(p);
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
