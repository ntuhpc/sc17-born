#include <vector>
#include <cmath>

using namespace std;

vector<float> compute_bound() {
  vector<float> bound_cpu(40);
  float _bcB=.0005;
  float _bcA=40;
  for(int i=0;i < 40; i++) bound_cpu[i]=expf(-_bcB*(_bcA-i));

  return bound_cpu;
}
