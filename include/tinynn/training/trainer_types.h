#pragma once

#include <tinynn/tinynn.h>  // SizeType

namespace tinynn {

template <class T>
struct StepResult {
  T loss = static_cast<T>(0);
  T accuracy = static_cast<T>(0);  // [0,1]
  SizeType batch_size = 0;
  SizeType num_correct = 0;
};

template <class T>
struct EpochResult {
  T loss = static_cast<T>(0);
  T accuracy = static_cast<T>(0);  // [0,1]
  SizeType num_samples = 0;
  SizeType num_correct = 0;
};

struct FitOptions {
  int start_epoch = 0;          // resume: first epoch index to run
  int epochs = 1;               // total epochs (exclusive upper bound)
};

}  // namespace tinynn
