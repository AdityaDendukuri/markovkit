#pragma once

#include "blas/matrix_ops.hpp"
#include "container/matrix.hpp"
#include "core/types.hpp"
#include <lapacke.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace rdme {

// Saye Algorithm 2 with the existing contiguous population sectors as blocks.
// Fit Lambda_i (T E)_i = E_i by least squares, T = W A W, E_0 = I.
// Dense matrices live only in occupied population-block positions. No sketch,
// propagator truncation, or normal equations are used. This is a small-block
// experiment: full population-block coefficients become expensive very quickly.
class BlockCascadingSmoother {
  using Dense = num::mat<double>;
  using Blocks = num::array<num::array<Dense>>;
  num::array<num::idx> offsets_;

  static Blocks multiply(const Blocks &a, const Blocks &b) {
    const auto count=a.size();
    Blocks result(count, num::array<Dense>(count));
    for (num::idx i=0;i<count;++i)
      for (num::idx k=0;k<count;++k) if (a[i][k].size())
        for (num::idx j=0;j<count;++j) if (b[k][j].size()) {
          auto &c=result[i][j];
          if (!c.size()) c=Dense(a[i][k].rows(),b[k][j].cols(),0.);
          num::blas::gemm(1.,a[i][k].data(),a[i][k].cols(),false,
                          b[k][j].data(),b[k][j].cols(),false,
                          1.,c.data(),c.cols(),c.rows(),c.cols(),a[i][k].cols());
        }
    return result;
  }

public:
  num::array<num::array<Dense>> coefficient;
  num::array<double> fitted_norm_squared;

  template<class Matrix, class Offsets>
  void build(const Matrix &a, const Offsets &offsets, int depth, double rcond=1e-8) {
    if (!(rcond>0. && rcond<1.) || !std::isfinite(rcond))
      throw std::invalid_argument("block cascade SVD cutoff must lie in (0,1)");
    if (depth<1 || depth>3) throw std::invalid_argument("block cascade depth must be 1..3");
    if (offsets.size()<2 || offsets.front()!=0 || offsets.back()!=a.rows())
      throw std::invalid_argument("block cascade offsets must partition the matrix");
    offsets_.assign(offsets.begin(),offsets.end());
    const auto count=offsets_.size()-1;
    std::uint64_t entries=0;
    for (num::idx i=0;i<count;++i) {
      if (offsets_[i+1]<=offsets_[i])
        throw std::invalid_argument("block cascade requires nonempty ordered blocks");
      const auto size=offsets_[i+1]-offsets_[i];
      if (size>1024)
        throw std::runtime_error("full block cascade limited to 1024 states per population block");
      entries+=size*size;
    }
    if (entries*depth*sizeof(double)>256ULL*1024*1024)
      throw std::runtime_error("block cascade coefficients exceed 256 MiB experiment limit");

    num::array<double> scale(a.rows());
    num::array<num::idx> owner(a.rows());
    Blocks t(count,num::array<Dense>(count)), e(count,num::array<Dense>(count));
    for (num::idx i=0;i<count;++i) {
      const auto size=offsets_[i+1]-offsets_[i];
      e[i][i]=Dense(size,size,0.);
      for (num::idx row=offsets_[i];row<offsets_[i+1];++row) {
        const double diagonal=a.value[a.diagonal[row]];
        if (!(diagonal>0.) || !std::isfinite(diagonal))
          throw std::invalid_argument("block cascade requires finite positive diagonal");
        scale[row]=1./std::sqrt(diagonal); owner[row]=i;
        e[i][i](row-offsets_[i],row-offsets_[i])=1.;
      }
    }
    for (num::idx row=0;row<a.rows();++row)
      for (auto entry=a.ptr[row];entry<a.ptr[row+1];++entry) {
        const auto column=a.column[entry];
        const auto i=owner[row], j=owner[column];
        if (!t[i][j].size())
          t[i][j]=Dense(offsets_[i+1]-offsets_[i],offsets_[j+1]-offsets_[j],0.);
        t[i][j](row-offsets_[i],column-offsets_[j])+=scale[row]*a.value[entry]*scale[column];
      }

    coefficient.clear(); fitted_norm_squared.clear();
    for (int stage=0;stage<depth;++stage) {
      auto g=multiply(t,e);
      num::array<Dense> factors(count);
      long double norm_squared=0.;
      for (num::idx i=0;i<count;++i) {
        const auto size=offsets_[i+1]-offsets_[i];
        num::idx columns=0;
        for (num::idx j=0;j<count;++j)
          if (g[i][j].size() || e[i][j].size()) columns+=offsets_[j+1]-offsets_[j];
        // Solve G_i^T Lambda_i^T = E_i^T for all block rows at once.
        // dgelsd handles zero/rank-deficient images in later stages.
        Dense design(columns,size,0.), rhs(std::max(columns,size),size,0.);
        num::idx start=0;
        for (num::idx j=0;j<count;++j) if (g[i][j].size() || e[i][j].size()) {
          const auto width=offsets_[j+1]-offsets_[j];
          for (num::idx r=0;r<size;++r) for (num::idx c=0;c<width;++c) {
            if (g[i][j].size()) design(start+c,r)=g[i][j](r,c);
            if (e[i][j].size()) rhs(start+c,r)=e[i][j](r,c);
          }
          start+=width;
        }
        num::array<double> singular_values(std::min(columns,size));
        lapack_int rank=0;
        const int status=LAPACKE_dgelsd(LAPACK_ROW_MAJOR,columns,size,size,
            design.data(),size,rhs.data(),size,singular_values.data(),rcond,&rank);
        if (status) throw std::runtime_error("block cascade least-squares solve failed");
        Dense lambda(size,size);
        for (num::idx r=0;r<size;++r) for (num::idx c=0;c<size;++c)
          lambda(r,c)=rhs(c,r);
        for (num::idx j=0;j<count;++j) if (g[i][j].size()) {
          auto &next=e[i][j];
          if (!next.size()) next=Dense(size,g[i][j].cols(),0.);
          num::blas::gemm(-1.,lambda.data(),size,false,g[i][j].data(),g[i][j].cols(),false,
                          1.,next.data(),next.cols(),size,next.cols(),size);
        }
        for (const auto &block:e[i])
          for (num::idx k=0;k<block.size();++k)
            norm_squared+=static_cast<long double>(block.data()[k])*block.data()[k];
        // Transform the fitted factors back to the original coordinates.
        for (num::idx r=0;r<size;++r) for (num::idx c=0;c<size;++c)
          lambda(r,c)*=scale[offsets_[i]+r]*scale[offsets_[i]+c];
        factors[i]=std::move(lambda);
      }
      coefficient.push_back(std::move(factors));
      fitted_norm_squared.push_back(static_cast<double>(norm_squared));
    }
  }

  template<class Matrix,class Vector>
  void apply(const Matrix &a,const Vector &b,Vector &x,Vector &work,bool reverse) const {
    for (num::idx visit=0;visit<coefficient.size();++visit) {
      const auto stage=reverse?coefficient.size()-1-visit:visit;
      a.apply(x,work); // Freeze the residual before updating any block.
      for (num::idx i=0;i<x.size();++i) work[i]=b[i]-work[i];
      for (num::idx block=0;block<offsets_.size()-1;++block) {
        const auto first=offsets_[block];
        const auto &lambda=coefficient[stage][block];
        // Matrix-vector product, viewed as a matrix product with one column.
        num::blas::gemm(1.,lambda.data(),lambda.cols(),reverse,
                        work.data()+first,1,false,1.,x.data()+first,1,
                        lambda.rows(),1,lambda.cols());
      }
    }
  }
  std::uint64_t bytes() const {
    std::uint64_t result=sizeof(num::idx)*offsets_.size();
    for (const auto &stage:coefficient) for (const auto &block:stage)
      result+=sizeof(double)*block.size();
    return result;
  }
};
} // namespace rdme
