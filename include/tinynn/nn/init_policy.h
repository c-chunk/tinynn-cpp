#pragma once

#include <cstdint>

namespace tinynn {

// Parameter initialization policy for trainable layers.
//
// Notes:
// - kDefault means "the layer's recommended default".
//   For example:
//     * Affine  : He normal
//     * Conv2d  : He normal
// - BatchNorm layers currently do not need this policy;
//   they use fixed initialization:
//     gamma=1, beta=0, running_mean=0, running_var=1.
enum class InitPolicy : std::uint8_t {
  kDefault = 0,
  kHeNormal,
  kXavierUniform,
  kZeros,
};

}  // namespace tinynn
