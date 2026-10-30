#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rdme {
// Scalar-block additive cascade, Saye (2026), Algorithm 2. Fit in the
// symmetrically prescaled system T = W A W, W_ii = A_ii^(-1/2):
//   lambda_si = <(T E)_i,E_i> / ||(T E)_i||^2,
//   E <- (I - diag(lambda_s) T) E, E_0 = I.
// Stored coefficients are W diag(lambda_s) W for the original system.
// row_cap=0 keeps the exact sparse error propagator. Positive row_cap
// sparsifies intermediate E only (always retaining its diagonal), so later
// stages solve the least-squares problem for that truncated propagator.
class CascadingSmoother {
  using Index = std::uint32_t;
  struct Sparse {
    std::vector<std::size_t> ptr{0};
    std::vector<Index> col;
    std::vector<double> val;
  };
public:
  std::vector<std::vector<double>> coefficient;
  std::vector<double> fitted_norm_squared;
  std::size_t max_propagator_nnz = 0;

  template<class Matrix>
  void build(const Matrix &a, int depth, std::size_t row_cap=0) {
    if (depth<1 || depth>4) throw std::invalid_argument("cascade depth must be 1..4");
    const std::size_t n=a.rows();
    std::vector<double> scale(n), product(n,0.), previous_row(n,0.);
    std::vector<Index> touched;
    std::vector<std::size_t> mark(n,n);
    Sparse e;
    for (std::size_t i=0;i<n;++i) {
      const double diagonal=a.value[a.diagonal[i]];
      if (!(diagonal>0.) || !std::isfinite(diagonal))
        throw std::runtime_error("cascade requires finite positive diagonal");
      scale[i]=1./std::sqrt(diagonal);
      e.col.push_back(static_cast<Index>(i)); e.val.push_back(1.); e.ptr.push_back(i+1);
    }
    coefficient.clear(); fitted_norm_squared.clear(); max_propagator_nnz=n;
    for (int stage=0;stage<depth;++stage) {
      std::vector<double> lambda(n);
      Sparse next;
      long double norm_squared=0.;
      for (std::size_t i=0;i<n;++i) {
        touched.clear();
        auto touch=[&](Index j) {
          if (mark[j]!=i) {mark[j]=i;product[j]=0.;previous_row[j]=0.;touched.push_back(j);}
        };
        for (std::size_t t=e.ptr[i];t<e.ptr[i+1];++t) {
          touch(e.col[t]); previous_row[e.col[t]]=e.val[t];
        }
        for (std::size_t t=a.ptr[i];t<a.ptr[i+1];++t) {
          const Index j=a.column[t];
          const double entry=scale[i]*a.value[t]*scale[j];
          for (std::size_t u=e.ptr[j];u<e.ptr[j+1];++u) {
            const Index k=e.col[u];touch(k);product[k]+=entry*e.val[u];
          }
        }
        long double numerator=0.,denominator=0.;
        for (Index j:touched) {
          numerator+=static_cast<long double>(previous_row[j])*product[j];
          denominator+=static_cast<long double>(product[j])*product[j];
        }
        const double fitted=denominator>0.?static_cast<double>(numerator/denominator):0.;
        if (!std::isfinite(fitted)) throw std::runtime_error("nonfinite cascade fit");
        lambda[i]=fitted*scale[i]*scale[i];
        for (Index j:touched) {
          previous_row[j]-=fitted*product[j];
          norm_squared+=static_cast<long double>(previous_row[j])*previous_row[j];
        }
        if (stage+1<depth) {
          if (row_cap && touched.size()>row_cap) {
            std::sort(touched.begin(),touched.end(),[&](Index j,Index k) {
              if (j==i || k==i) return j==i && k!=i;
              const double x=std::abs(previous_row[j]),y=std::abs(previous_row[k]);
              return x!=y?x>y:j<k;
            });
            touched.resize(row_cap);
          }
          std::sort(touched.begin(),touched.end());
          for (Index j:touched) if (previous_row[j]!=0.) {
            next.col.push_back(j);next.val.push_back(previous_row[j]);
          }
          next.ptr.push_back(next.col.size());
          if (next.col.size()>20000000)
            throw std::runtime_error("cascade propagator exceeds 20M entries; use --cascade-row-cap");
        }
      }
      coefficient.push_back(std::move(lambda));
      fitted_norm_squared.push_back(static_cast<double>(norm_squared));
      if (stage+1<depth) {
        max_propagator_nnz=std::max(max_propagator_nnz,next.col.size());
        e=std::move(next);
        // A fresh stage must not reuse the previous row stamps.
        std::fill(mark.begin(),mark.end(),n);
      }
    }
  }

  template<class Matrix,class Vector>
  void apply(const Matrix &a,const Vector &b,Vector &x,Vector &work,
             bool reverse_stages) const {
    for (std::size_t visit=0;visit<coefficient.size();++visit) {
      const std::size_t stage=reverse_stages?coefficient.size()-1-visit:visit;
      a.apply(x,work);
      for (std::size_t i=0;i<x.size();++i)
        x[i]+=coefficient[stage][i]*(b[i]-work[i]);
    }
  }
  std::uint64_t bytes() const {
    std::uint64_t count=0;for (const auto &v:coefficient)count+=8*v.size();return count;
  }
};
} // namespace rdme
