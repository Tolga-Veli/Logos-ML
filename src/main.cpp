#include <cblas.h>
#include <iostream>

int main() {
  float A[] = {1, 2, 3, 4};
  float B[] = {5, 6, 7, 8};
  float C[4]{};

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, 1.0f, A, 2, B,
              2, 0.0f, C, 2);

  std::cout << C[0] << std::endl;
}
