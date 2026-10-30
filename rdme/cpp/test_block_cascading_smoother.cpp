#include "block_cascading_smoother.hpp"
#include "cascading_smoother.hpp"
#include <iostream>
#include <numeric>

struct Matrix {
  num::array<num::idx> ptr{0}, column, diagonal;
  num::array<double> value;
  num::idx rows() const {return diagonal.size();}
  void apply(const num::vec<double> &x,num::vec<double> &y) const {
    for (num::idx i=0;i<rows();++i) {
      y[i]=0.;
      for (auto k=ptr[i];k<ptr[i+1];++k) y[i]+=value[k]*x[column[k]];
    }
  }
};
void require(bool ok,const char *message) {if (!ok) throw std::runtime_error(message);}
Matrix from_dense(const num::array<double> &a,num::idx n) {
  Matrix m; m.diagonal.resize(n);
  for (num::idx i=0;i<n;++i) {
    for (num::idx j=0;j<n;++j) if (a[i*n+j]!=0.) {
      if (i==j) m.diagonal[i]=m.value.size();
      m.column.push_back(j); m.value.push_back(a[i*n+j]);
    }
    m.ptr.push_back(m.value.size());
  }
  return m;
}
int main() {
  try {
    const num::idx n=4;
    const num::array<double> a{4,-1,0,0,-2,5,-1,0,0,-3,6,-1,-.2,0,-2,3};
    auto m=from_dense(a,n);
    const num::array<num::idx> offsets{0,2,4};
    rdme::BlockCascadingSmoother block; block.build(m,offsets,2);
    num::mat<double> e(n,n,0.),t(n,n),g(n,n),next(n,n);
    num::array<double> scale(n);
    for (num::idx i=0;i<n;++i) {e(i,i)=1.;scale[i]=1./std::sqrt(a[i*n+i]);}
    for (num::idx i=0;i<n;++i) for (num::idx j=0;j<n;++j)
      t(i,j)=scale[i]*a[i*n+j]*scale[j];
    double previous_norm=n;
    for (num::idx stage=0;stage<2;++stage) {
      for (num::idx i=0;i<n;++i) for (num::idx j=0;j<n;++j) {
        g(i,j)=0.;
        for (num::idx k=0;k<n;++k) g(i,j)+=t(i,k)*e(k,j);
      }
      for (num::idx b=0;b<2;++b) {
        const auto &lambda=block.coefficient[stage][b];
        for (num::idx i=0;i<2;++i) for (num::idx j=0;j<n;++j) {
          const auto row=2*b+i;
          next(row,j)=e(row,j);
          for (num::idx k=0;k<2;++k)
            next(row,j)-=lambda(i,k)/(scale[row]*scale[2*b+k])*g(2*b+k,j);
        }
        // Least-squares residual is orthogonal to every design column.
        for (num::idx i=0;i<2;++i) for (num::idx k=0;k<2;++k) {
          double inner=0.;
          for (num::idx j=0;j<n;++j) inner+=next(2*b+i,j)*g(2*b+k,j);
          require(std::abs(inner)<1e-11,"block least-squares orthogonality failed");
        }
      }
      double norm=0.;for (num::idx i=0;i<next.size();++i) norm+=next.data()[i]*next.data()[i];
      require(norm<=previous_norm+1e-12,"block cascade increased Frobenius norm");
      require(std::abs(norm-block.fitted_norm_squared[stage])<1e-11,"incorrect block fit norm");
      previous_norm=norm;e=next;
    }
    const num::vec<double> initial{.3,-.2,.7,1.1},rhs{.1,.4,-.3,.2};
    for (bool reverse:{false,true}) {
      auto x=initial, ref=initial;num::vec<double> work(n),image(n);
      block.apply(m,rhs,x,work,reverse);
      for (num::idx visit=0;visit<2;++visit) {
        const auto stage=reverse?1-visit:visit;
        m.apply(ref,image);
        for (num::idx b=0;b<2;++b) for (num::idx i=0;i<2;++i)
          for (num::idx j=0;j<2;++j) {
            const auto &lambda=block.coefficient[stage][b];
            ref[2*b+i]+=(reverse?lambda(j,i):lambda(i,j))*(rhs[2*b+j]-image[2*b+j]);
          }
      }
      for (num::idx i=0;i<n;++i) require(std::abs(x[i]-ref[i])<1e-12,"block application or transpose order failed");
    }
    // Singleton blocks must reproduce the scalar additive cascade.
    rdme::BlockCascadingSmoother singletons;singletons.build(m,num::array<num::idx>{0,1,2,3,4},3);
    rdme::CascadingSmoother scalar;scalar.build(m,3);
    for (num::idx stage=0;stage<3;++stage) for (num::idx i=0;i<n;++i)
      require(std::abs(singletons.coefficient[stage][i](0,0)-scalar.coefficient[stage][i])<1e-11,"singleton block differs from scalar fit");
    // Rank-deficient later fits must remain finite after an exact block solve.
    auto diagonal=from_dense({2.,0.,0.,7.},2);
    rdme::BlockCascadingSmoother eliminated;eliminated.build(diagonal,num::array<num::idx>{0,2},2);
    num::vec<double> b{1.,1.},x(2,0.),work(2);
    eliminated.apply(diagonal,b,x,work,false);
    require(std::abs(x[0]-.5)<1e-12 && std::abs(x[1]-1./7)<1e-12,"exact diagonal block solve failed");
    for (const auto &stage:eliminated.coefficient) for (const auto &factor:stage)
      for (num::idx i=0;i<factor.size();++i) require(std::isfinite(factor.data()[i]),"nonfinite rank-deficient fit");
    std::cout<<"block cascade: least squares, scaling, norms, singleton equivalence, transpose order, and rank deficiency passed\n";
  } catch (const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
