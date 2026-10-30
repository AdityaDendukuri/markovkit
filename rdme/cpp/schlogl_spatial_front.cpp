// Exact direct Gillespie simulation of a reflecting one- or two-dimensional RDME.
// Independent front demonstration; this is not a multigrid/FSP solve.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct Fenwick {
  std::vector<double> tree, values;
  explicit Fenwick(int size):tree(size+1,0.),values(size,0.){}
  void set(int i,double value){
    double change=value-values[i];values[i]=value;
    for(int j=i+1;j<int(tree.size());j+=j&-j)tree[j]+=change;
  }
  double sum()const{double s=0.;for(int i=int(values.size());i;i-=i&-i)s+=tree[i];return s;}
  int select(double target)const{
    int index=0,bit=1;
    while(2*bit<int(tree.size()))bit*=2;
    for(;bit;bit/=2){int next=index+bit;if(next<int(tree.size())&&tree[next]<=target){index=next;target-=tree[next];}}
    return std::min(index,int(values.size())-1);
  }
};
struct Model {
  int cells=64,seeded=16,paths=64,low=2,high=10,dimension=1;
  int voxels()const{return dimension==1?cells:cells*cells;}
  int channels()const{return 2+2*dimension;}
  double volume=20.,jump=1.,final=300.,interval=2.,reaction_scale=1.;
  double c1=.17,c2=.03,c3=.5,c4=.4;
  std::uint64_t seed=20261021;
  std::string prefix;
  bool marginals=false,totals_only=false;
};
struct Simulation {
  const Model &m;
  std::vector<int> n;
  Fenwick rates;
  std::mt19937_64 random;
  std::uint64_t events=0;
  explicit Simulation(const Model &model,std::uint64_t seed):m(model),n(m.voxels()),rates(m.channels()*m.voxels()),random(seed){
    for(int v=0;v<m.voxels();++v)n[v]=int(std::llround(m.volume*(v%m.cells<m.seeded?m.high:m.low)));
    for(int v=0;v<m.voxels();++v)update(v);
  }
  double uniform(){return (double(random()>>11)+.5)*0x1.0p-53;}
  int neighbor(int v,int direction)const{
    const int x=v%m.cells,y=v/m.cells;
    if(direction==0)return x>0?v-1:-1;
    if(direction==1)return x+1<m.cells?v+1:-1;
    if(direction==2)return y>0?v-m.cells:-1;
    return y+1<m.cells?v+m.cells:-1;
  }
  void update(int v){
    const double x=n[v],w=m.volume;
    // Same mass-action/factorial convention as rdme_full_multigrid.
    rates.set(m.channels()*v,m.reaction_scale*(m.c3*w+m.c1*x*(x-1)/(2*w)));
    rates.set(m.channels()*v+1,m.reaction_scale*(m.c4*x+m.c2*x*(x-1)*(x-2)/(6*w*w)));
    for(int d=0;d<2*m.dimension;++d)rates.set(m.channels()*v+2+d,neighbor(v,d)>=0?m.jump*x:0.);
  }
  void event(double total){
    const int channel=rates.select(uniform()*total),v=channel/m.channels(),type=channel%m.channels();
    if(type==0)++n[v];
    else if(type==1)--n[v];
    else {const int w=neighbor(v,type-2);if(w<0)throw std::runtime_error("boundary jump");--n[v];++n[w];update(w);}
    if(n[v]<0)throw std::runtime_error("negative population");
    update(v);++events;
  }
};
int main(int argc,char**argv){try{
  Model m;
  for(int i=1;i<argc;++i){std::string arg=argv[i];if(i+1==argc)throw std::runtime_error("missing value");std::string value=argv[++i];
    if(arg=="--dimension")m.dimension=std::stoi(value);else if(arg=="--cells")m.cells=std::stoi(value);else if(arg=="--seeded")m.seeded=std::stoi(value);
    else if(arg=="--totals-only")m.totals_only=std::stoi(value)!=0;
    else if(arg=="--marginals")m.marginals=std::stoi(value)!=0;
    else if(arg=="--paths")m.paths=std::stoi(value);else if(arg=="--volume")m.volume=std::stod(value);
    else if(arg=="--jump")m.jump=std::stod(value);else if(arg=="--final-time")m.final=std::stod(value);
    else if(arg=="--interval")m.interval=std::stod(value);else if(arg=="--seed")m.seed=std::stoull(value);
    else if(arg=="--c1")m.c1=std::stod(value);
    else if(arg=="--c2")m.c2=std::stod(value);
    else if(arg=="--c3")m.c3=std::stod(value);
    else if(arg=="--c4")m.c4=std::stod(value);
    else if(arg=="--reaction-scale")m.reaction_scale=std::stod(value);
    else if(arg=="--low")m.low=std::stoi(value);else if(arg=="--high")m.high=std::stoi(value);
    else if(arg=="--output-prefix")m.prefix=value;else throw std::runtime_error("unknown argument");
  }
  if((m.dimension!=1&&m.dimension!=2)||m.cells<2||m.cells>1024||m.seeded<0||m.seeded>m.cells||m.paths<1||m.low<0||m.high<0||
     !(m.volume>0)||!(m.jump>=0)||!(m.final>0)||!(m.interval>0)||!(m.reaction_scale>=0)||m.prefix.empty()||m.c1<0||m.c2<0||m.c3<0||m.c4<0)throw std::runtime_error("invalid parameters");
  const int snapshots=int(std::llround(m.final/m.interval))+1;
  if(std::abs((snapshots-1)*m.interval-m.final)>1e-8)throw std::runtime_error("final time must be a multiple of interval");
  const int size=snapshots*m.voxels();
  std::vector<double> sums(size,0.),squares(size,0.);std::vector<int> high(size,0);
  std::vector<std::map<int,int>> histograms(m.marginals?size:0);
  std::vector<std::map<int,std::uint64_t>> total_histograms(snapshots);
  std::ofstream sample(m.prefix+"_sample.csv"),fronts(m.prefix+"_fronts.csv");
  if(!sample||!fronts)throw std::runtime_error("cannot open output");
  sample<<"time,voxel,count,concentration\n"<<std::setprecision(16);
  fronts<<"path,time,high_voxels,left_connected_high_voxels,total_molecules\n";
  std::uint64_t all_events=0;
  for(int path=0;path<m.paths;++path){
    Simulation s(m,m.seed+0x9e3779b97f4a7c15ULL*std::uint64_t(path));
    double next=0.;double total=s.rates.sum();next=total>0?-std::log(s.uniform())/total:INFINITY;
    for(int k=0;k<snapshots;++k){const double time=k*m.interval;
      while(next<=time){s.event(total);total=s.rates.sum();next+=total>0?-std::log(s.uniform())/total:INFINITY;}
      int high_count=0,connected=0,molecules=0;
      for(int v=0;v<m.voxels();++v){const double x=s.n[v]/m.volume;const int j=k*m.voxels()+v;
        if(m.marginals)++histograms[j][s.n[v]];
        sums[j]+=x;squares[j]+=x*x;high[j]+=x>=5.;high_count+=x>=5.;molecules+=s.n[v];
        if(v==connected&&x>=5.)++connected;
        if(path==0)sample<<time<<','<<v<<','<<s.n[v]<<','<<x<<'\n';
      }
      if(m.dimension==2){
        connected=0;std::vector<unsigned char> seen(m.voxels(),0);std::vector<int> stack;
        for(int v=0;v<m.voxels();v+=m.cells)if(s.n[v]/m.volume>=5.){stack.push_back(v);seen[v]=1;}
        while(!stack.empty()){const int v=stack.back();stack.pop_back();++connected;
          for(int d=0;d<4;++d){const int w=s.neighbor(v,d);if(w>=0&&!seen[w]&&s.n[w]/m.volume>=5.){seen[w]=1;stack.push_back(w);}}
        }
      }
      ++total_histograms[k][molecules];
      if(!m.totals_only)fronts<<path<<','<<time<<','<<high_count<<','<<connected<<','<<molecules<<'\n';
    }
    all_events+=s.events;
    if(m.paths<=256||(path+1)%1000==0)std::cerr<<"path="<<path+1<<" events="<<s.events<<'\n';
  }
  std::ofstream mean(m.prefix+"_ensemble.csv");if(!mean)throw std::runtime_error("cannot open ensemble output");
  mean<<"time,voxel,mean_concentration,standard_error,high_probability,high_probability_se\n"<<std::setprecision(16);
  for(int k=0;k<snapshots;++k)for(int v=0;v<m.voxels();++v){const int j=k*m.voxels()+v;const double avg=sums[j]/m.paths,p=double(high[j])/m.paths;
    const double variance=m.paths>1?std::max(0.,(squares[j]-sums[j]*sums[j]/m.paths)/(m.paths-1)):0.;
    mean<<k*m.interval<<','<<v<<','<<avg<<','<<std::sqrt(variance/m.paths)<<','<<p<<','<<std::sqrt(p*(1-p)/m.paths)<<'\n';
  }
  {
    std::ofstream out(m.prefix+"_total_marginals.csv");if(!out)throw std::runtime_error("cannot open total marginal output");
    out<<"time,count,frequency,probability\n"<<std::setprecision(16);
    for(int k=0;k<snapshots;++k)for(const auto &[count,frequency]:total_histograms[k])
      out<<k*m.interval<<','<<count<<','<<frequency<<','<<double(frequency)/m.paths<<'\n';
  }
  if(m.marginals){
    std::ofstream out(m.prefix+"_marginals.csv");if(!out)throw std::runtime_error("cannot open marginal output");
    out<<"time,voxel,count,frequency,probability\n"<<std::setprecision(16);
    for(int k=0;k<snapshots;++k)for(int v=0;v<m.voxels();++v)
      for(const auto &[count,frequency]:histograms[k*m.voxels()+v])
        out<<k*m.interval<<','<<v<<','<<count<<','<<frequency<<','<<double(frequency)/m.paths<<'\n';
  }
  std::cout<<"paths,events,cells,volume,jump,final_time,seed\n"<<m.paths<<','<<all_events<<','<<m.cells<<','<<m.volume<<','<<m.jump<<','<<m.final<<','<<m.seed<<'\n';
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
