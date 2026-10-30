#include "cascading_smoother.hpp"
#include <iostream>
#include <numeric>

struct Matrix {
  std::vector<std::size_t> ptr{0},column,diagonal;
  std::vector<double> value;
  std::size_t rows()const{return diagonal.size();}
  void apply(const std::vector<double>&x,std::vector<double>&y)const {
    for(std::size_t i=0;i<rows();++i){y[i]=0.;for(auto t=ptr[i];t<ptr[i+1];++t)y[i]+=value[t]*x[column[t]];}
  }
};
void require(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
Matrix from_dense(const std::vector<double>&a,std::size_t n){
  Matrix m;m.diagonal.resize(n);
  for(std::size_t i=0;i<n;++i){
    for(std::size_t j=0;j<n;++j)if(a[i*n+j]!=0.){
      if(i==j)m.diagonal[i]=m.value.size();m.column.push_back(j);m.value.push_back(a[i*n+j]);
    }m.ptr.push_back(m.value.size());
  }return m;
}
// Reproduce all four additive coefficients in Saye's Table 1, not just the
// first-stage Jacobi weight. The table prints coefficients to two decimals.
void test_published_poisson_coefficients() {
  for (int dimension : {2,3}) {
    const std::size_t side=10, n=dimension==2?100:1000;
    Matrix a; a.diagonal.resize(n);
    for (std::size_t i=0;i<n;++i) {
      std::vector<std::pair<std::size_t,double>> row{{i,1.}};
      std::size_t stride=1;
      for (int axis=0;axis<dimension;++axis,stride*=side) {
        const auto coordinate=(i/stride)%side;
        const auto next=coordinate+1<side?i+stride:i-(side-1)*stride;
        const auto previous=coordinate>0?i-stride:i+(side-1)*stride;
        row.push_back({next,-1./(2*dimension)});
        row.push_back({previous,-1./(2*dimension)});
      }
      std::sort(row.begin(),row.end());
      for (auto [j,value]:row) {
        if (j==i) a.diagonal[i]=a.value.size();
        a.column.push_back(j); a.value.push_back(value);
      }
      a.ptr.push_back(a.value.size());
    }
    const std::vector<double> published=dimension==2
      ?std::vector<double>{.80,.86,.91,.92}
      :std::vector<double>{.86,.88,.90,.93};
    rdme::CascadingSmoother fit; fit.build(a,4);
    for (std::size_t stage=0;stage<published.size();++stage)
      for (double value:fit.coefficient[stage])
        require(std::abs(value-published[stage])<.005,
                "periodic Poisson coefficient disagrees with Saye Table 1");
  }
}
int main(){
 try {
  test_published_poisson_coefficients();
  // A nonsymmetric, nonuniform-diagonal problem checks the scaling and the
  // sparse fitted propagators against an independent dense matrix calculation.
  const std::size_t n=4;
  const std::vector<double>a={4,-1,0,0,-2,5,-1,0,0,-3,6,-1,-.2,0,-2,3};
  const auto m=from_dense(a,n);
  rdme::CascadingSmoother c;c.build(m,3);
  std::vector<double>t(n*n),e(n*n,0.),g(n*n),next(n*n),scale(n);
  for(std::size_t i=0;i<n;++i){scale[i]=1/std::sqrt(a[i*n+i]);e[i*n+i]=1.;}
  for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j)t[i*n+j]=scale[i]*a[i*n+j]*scale[j];
  double previous_norm=n;
  for(int stage=0;stage<3;++stage){
    std::fill(g.begin(),g.end(),0.);
    for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j)for(std::size_t k=0;k<n;++k)g[i*n+j]+=t[i*n+k]*e[k*n+j];
    for(std::size_t i=0;i<n;++i){
      double numer=0.,denom=0.;for(std::size_t j=0;j<n;++j){numer+=e[i*n+j]*g[i*n+j];denom+=g[i*n+j]*g[i*n+j];}
      const double lambda=numer/denom;
      require(std::abs(c.coefficient[stage][i]-lambda/a[i*n+i])<2e-13,"sparse fit differs from dense fit");
      double orthogonal=0.;for(std::size_t j=0;j<n;++j){next[i*n+j]=e[i*n+j]-lambda*g[i*n+j];orthogonal+=next[i*n+j]*g[i*n+j];}
      require(std::abs(orthogonal)<2e-13,"least-squares residual is not orthogonal");
    }
    double norm=std::inner_product(next.begin(),next.end(),next.begin(),0.);
    require(norm<=previous_norm+1e-13,"exact cascade does not reduce Frobenius norm");
    require(std::abs(norm-c.fitted_norm_squared[stage])<2e-13,"reported Frobenius norm differs");
    previous_norm=norm;e=next;
  }
  std::vector<double>x={.3,-.2,.7,1.1},initial=x,b(n,0.),work(n);
  c.apply(m,b,x,work,false);
  for(std::size_t i=0;i<n;++i){double expected=0.;for(std::size_t j=0;j<n;++j)expected+=scale[i]*e[i*n+j]*initial[j]/scale[j];
    require(std::abs(x[i]-expected)<3e-13,"unscaled application differs from fitted propagator");}
  // Independently reproduce the reverse-stage application in original coordinates.
  x=initial;std::vector<double>ref=initial;
  c.apply(m,b,x,work,true);
  for(int stage=2;stage>=0;--stage){std::vector<double>image(n,0.);for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j)image[i]+=a[i*n+j]*ref[j];
    for(std::size_t i=0;i<n;++i)ref[i]-=c.coefficient[stage][i]*image[i];}
  for(std::size_t i=0;i<n;++i)require(std::abs(x[i]-ref[i])<2e-13,"reverse application differs");
  // Textbook five- and seven-point unit-diagonal Poisson stencil weights.
  for(int degree:{4,6}){
    std::size_t size=degree+1;std::vector<double>d(size*size,0.);
    for(std::size_t i=0;i<size;++i)d[i*size+i]=1.;
    for(int j=1;j<=degree;++j)d[j]=d[j*size]=-1./degree;
    auto poisson=from_dense(d,size);rdme::CascadingSmoother fit;fit.build(poisson,1);
    require(std::abs(fit.coefficient[0][0]-double(degree)/(degree+1))<2e-14,"incorrect Poisson damping coefficient");
  }
  // Exact first-stage elimination leaves zero-image rows in later fits.
  auto diagonal=from_dense({2.,0.,0.,7.},2);
  rdme::CascadingSmoother eliminated;eliminated.build(diagonal,3);
  require(std::abs(eliminated.coefficient[0][0]-.5)<1e-14,"diagonal solve coefficient");
  for(int stage=1;stage<3;++stage)for(double v:eliminated.coefficient[stage])
    require(std::isfinite(v),"zero-image row produced nonfinite coefficient");
  bool rejected=false;try{rdme::CascadingSmoother bad;bad.build(m,0);}catch(const std::invalid_argument&){rejected=true;}
  require(rejected,"invalid cascade depth was accepted");
  // Truncation is deterministic, and never changes the first-stage fit.
  rdme::CascadingSmoother truncated;truncated.build(m,3,2);
  for(std::size_t i=0;i<n;++i)require(truncated.coefficient[0][i]==c.coefficient[0][i],"truncation changed first stage");
  require(truncated.max_propagator_nnz<=n*2,"row cap was not respected");
  std::cout<<"cascade: dense fit, scaling, application order, Poisson weights, and row cap passed\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
