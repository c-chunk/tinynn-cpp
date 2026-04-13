#include <cassert>
#include <iostream>

#include <tinynn/nn/layers/adaptive_avg_pool2d.h>
#include <tinynn/tensor/tensor.h>
#include <tinynn/tensor/tensor_view.h>

using tinynn::AdaptiveAvgPool2d;
using tinynn::Shape;
using tinynn::Tensor;
using tinynn::TensorView;

int main() {
  using T = float;

  // input tensor
  // shape: [1,1,4,4]
  Tensor<T> x(Shape{{1,1,4,4}});

  // fill with 1..16
  for (int i = 0; i < 16; ++i) {
    x.data()[i] = static_cast<T>(i + 1);
  }

  // pool -> (2,2)
  AdaptiveAvgPool2d<T> pool(2,2);

  Tensor<T> y(Shape{{1,1,2,2}});

  pool.forward(x.view(), y.view());

  std::cout << "Forward output:\n";

  for (int i = 0; i < y.size(); ++i) {
    std::cout << y.data()[i] << " ";
  }
  std::cout << "\n";

  // expected:
  // block averages
  //
  // [1 2 | 3 4]
  // [5 6 | 7 8]
  // -------------
  // [9 10 | 11 12]
  // [13 14 | 15 16]
  //
  // =>
  //
  // (1+2+5+6)/4   (3+4+7+8)/4
  // (9+10+13+14)/4 (11+12+15+16)/4

  assert(y(0,0,0,0) == (1+2+5+6)/4.0f);
  assert(y(0,0,0,1) == (3+4+7+8)/4.0f);
  assert(y(0,0,1,0) == (9+10+13+14)/4.0f);
  assert(y(0,0,1,1) == (11+12+15+16)/4.0f);

  // backward test

  Tensor<T> dy(Shape{{1,1,2,2}});
  Tensor<T> dx(Shape{{1,1,4,4}});

  for (int i = 0; i < dy.size(); ++i) {
    dy.data()[i] = 1.0f;
  }

  pool.backward(dy.view(), dx.view());

  std::cout << "Backward dx:\n";

  for (int i = 0; i < dx.size(); ++i) {
    std::cout << dx.data()[i] << " ";
  }
  std::cout << "\n";

  // each input cell should receive 1/4
  for (int i = 0; i < dx.size(); ++i) {
    assert(dx.data()[i] == 0.25f);
  }

  std::cout << "[OK] AdaptiveAvgPool2d minimal test passed\n";

  return 0;
}
