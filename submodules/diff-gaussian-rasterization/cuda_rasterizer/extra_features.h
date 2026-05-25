#ifndef CUDA_RASTERIZER_EXTRA_FEATURES_H_INCLUDED
#define CUDA_RASTERIZER_EXTRA_FEATURES_H_INCLUDED

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "rasterizer.h"
#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include "auxiliary.h"
#include "stopthepop/hierarchical_render.cuh"
#include "stopthepop/resorted_render.cuh"

template <int32_t N_EXTRA_FEATURES> class ExtraFeaturesBlender {
  // use a class for less clutter
public:
  void blendBackward(const dim3 grid, dim3 block, const uint2 *ranges,
                     const CudaRasterizer::SplattingSettings splatting_settings,
                     const uint32_t *point_list, int W, int H, float focal_x,
                     float focal_y, const float *bg_color,
                     const float *extra_features, const float2 *means2D,
                     const float4 *cov3D_inv, const float *projmatrix_inv,
                     const glm::vec3 *cam_pos, const float4 *conic_opacity,
                     const float *view2gaussian, const float *viewmatrix,
                     const float *final_Ts, const uint32_t *n_contrib,
                     const float *pixel_colors, const float *dL_dpixels,
                    float3* dL_dmean2D, float4* dL_dconic2D, float* dL_dopacity, float* dL_dview2gaussian,
                     float *dL_dextra_features) {
    using namespace CudaRasterizer;
#define CALL_KBUFFER(WINDOW) throw std::runtime_error("Not Impl")

    if (splatting_settings.sort_settings.sort_mode == SortMode::GLOBAL) {
      throw std::runtime_error("Not Impl");
    } else if (splatting_settings.sort_settings.sort_mode ==
               SortMode::PER_PIXEL_KBUFFER) {
      int window_size = splatting_settings.sort_settings.queue_sizes.per_pixel;
      if (window_size <= 1)
        CALL_KBUFFER(1);
      else if (window_size <= 2)
        CALL_KBUFFER(2);
      else if (window_size <= 4)
        CALL_KBUFFER(4);
      else if (window_size <= 8)
        CALL_KBUFFER(8);
      else if (window_size <= 12)
        CALL_KBUFFER(12);
      else if (window_size <= 16)
        CALL_KBUFFER(16);
      else if (window_size <= 20)
        CALL_KBUFFER(20);
      else
        CALL_KBUFFER(24);
      return;
    } else if (splatting_settings.sort_settings.sort_mode ==
               SortMode::PER_PIXEL_FULL) {
      throw std::runtime_error(
          "Backward not supported for full per-pixel sort");
    } else if (splatting_settings.sort_settings.sort_mode ==
               SortMode::HIERARCHICAL) {
#define CALL_HIER_DETACHALPHA(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE,   \
                              DETACH_ALPHA)                                    \
  sortGaussiansRayHierarchicalCUDA_blendExtraFeaturesBackward<                 \
      NUM_CHANNELS, N_EXTRA_FEATURES, HEAD_QUEUE_SIZE, MID_QUEUE_SIZE,         \
      HIER_CULLING><<<grid, {16, 4, 4}>>>(                                     \
      ranges, point_list, W, H, focal_x, focal_y,                              \
      splatting_settings.far_plane, view2gaussian, means2D, cov3D_inv,         \
      projmatrix_inv, (float3 *)cam_pos, conic_opacity, bg_color,              \
      extra_features, final_Ts, n_contrib, pixel_colors, dL_dpixels,           \
      dL_dmean2D, dL_dconic2D, dL_dopacity, dL_dview2gaussian,dL_dextra_features)

#define CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE)               \
  if (splatting_settings.detach_alpha) {                                       \
    CALL_HIER_DETACHALPHA(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE,       \
                          true);                                               \
  } else {                                                                     \
    CALL_HIER_DETACHALPHA(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE,       \
                          false);                                              \
  }

#ifndef STOPTHEPOP_FASTBUILD
#define CALL_HIER_HEAD(HIER_CULLING, MID_QUEUE_SIZE)                           \
  switch (splatting_settings.sort_settings.queue_sizes.per_pixel) {            \
  case 4: {                                                                    \
    CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, 4);                                \
    break;                                                                     \
  }                                                                            \
  case 8: {                                                                    \
    CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, 8);                                \
    break;                                                                     \
  }                                                                            \
  case 12: {                                                                   \
    CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, 12);                               \
    break;                                                                     \
  }                                                                            \
  case 16: {                                                                   \
    CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, 16);                               \
    break;                                                                     \
  }                                                                            \
  default: {                                                                   \
    throw std::runtime_error(                                                  \
        "Not supported head queue size " +                                     \
        std::to_string(                                                        \
            splatting_settings.sort_settings.queue_sizes.per_pixel));          \
  }                                                                            \
  }

#define CALL_HIER_MID(HIER_CULLING)                                            \
  switch (splatting_settings.sort_settings.queue_sizes.tile_2x2) {             \
  case 8: {                                                                    \
    CALL_HIER_HEAD(HIER_CULLING, 8);                                           \
    break;                                                                     \
  }                                                                            \
  case 12: {                                                                   \
    CALL_HIER_HEAD(HIER_CULLING, 12);                                          \
    break;                                                                     \
  }                                                                            \
  case 20: {                                                                   \
    CALL_HIER_HEAD(HIER_CULLING, 20);                                          \
    break;                                                                     \
  }                                                                            \
  default: {                                                                   \
    throw std::runtime_error(                                                  \
        "Not supported mid queue size " +                                      \
        std::to_string(                                                        \
            splatting_settings.sort_settings.queue_sizes.tile_2x2));           \
  }                                                                            \
  }
#else
#define CALL_HIER_HEAD(HIER_CULLING, MID_QUEUE_SIZE)                           \
  switch (splatting_settings.sort_settings.queue_sizes.per_pixel) {            \
  case 4: {                                                                    \
    CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, 4);                                \
    break;                                                                     \
  }                                                                            \
  default: {                                                                   \
    throw std::runtime_error(                                                  \
        "Not supported head queue size " +                                     \
        std::to_string(                                                        \
            splatting_settings.sort_settings.queue_sizes.per_pixel));          \
  }                                                                            \
  }

#define CALL_HIER_MID(HIER_CULLING)                                            \
  switch (splatting_settings.sort_settings.queue_sizes.tile_2x2) {             \
  case 8: {                                                                    \
    CALL_HIER_HEAD(HIER_CULLING, 8);                                           \
    break;                                                                     \
  }                                                                            \
  default: {                                                                   \
    throw std::runtime_error(                                                  \
        "Not supported mid queue size " +                                      \
        std::to_string(                                                        \
            splatting_settings.sort_settings.queue_sizes.tile_2x2));           \
  }                                                                            \
  }
#endif // STOPTHEPOP_FASTBUILD

      if (splatting_settings.culling_settings.hierarchical_4x4_culling) {
        CALL_HIER_MID(true);
      } else {
        CALL_HIER_MID(false);
      }

#undef CALL_HIER_MID
#undef CALL_HIER_HEAD
#undef CALL_HIER
#undef CALL_HIER_DETACHALPHA
    }
  };

  void blendForward(const dim3 grid, dim3 block, const uint2 *ranges,
                    const CudaRasterizer::SplattingSettings splatting_settings,
                    const uint32_t *point_list, int W, int H, float focal_x,
                    float focal_y, const float *extra_features,
                    const float2 *means2D, const float *view2gaussian,
                    const float *means3D, const float4 *cov3D_inv,
                    const float *projmatrix_inv, const glm::vec3 *cam_pos,
                    const float *colors, // = feature_ptr
                    const float *depths, const float4 *conic_opacity,
                    const float *bg_color,
                    CudaRasterizer::DebugVisualizationData &debugVisualization,
                    float *out_color) {

    using namespace CudaRasterizer;
    if (splatting_settings.sort_settings.sort_mode == SortMode::GLOBAL) {
      throw std::runtime_error(
          "NOT IMPLMENETED BLEND_EXTRA_FEATURES FOR THIS SORT MODE");
    } else {

#define CALL_HIER_DEBUG(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE,         \
                        EXACT_DEPTH, DEBUG)                                    \
  sortGaussiansRayHierarchicalCUDA_blendExtraFeaturesForward<                  \
      NUM_CHANNELS, N_EXTRA_FEATURES, HEAD_QUEUE_SIZE, MID_QUEUE_SIZE,         \
      HIER_CULLING, EXACT_DEPTH, DEBUG><<<grid, {16, 4, 4}>>>(                 \
      ranges, point_list, W, H, focal_x, focal_y,                              \
      splatting_settings.far_plane, extra_features, view2gaussian, means2D,    \
      cov3D_inv, projmatrix_inv, (float3 *)cam_pos, colors, conic_opacity,     \
      bg_color, debugVisualization.type, out_color)

#define CALL_HIER_EXACT_DEPTH(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE,   \
                              EXACT_DEPTH)                                     \
  if (debugVisualization.type == DebugVisualization::Opacity) {                \
    CALL_HIER_DEBUG(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE,             \
                    EXACT_DEPTH, true);                                        \
  } else {                                                                     \
    CALL_HIER_DEBUG(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE,             \
                    EXACT_DEPTH, false);                                       \
  }

#define CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE)               \
  if (splatting_settings.exact_depth) {                                        \
    CALL_HIER_EXACT_DEPTH(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE,       \
                          true);                                               \
  } else {                                                                     \
    CALL_HIER_EXACT_DEPTH(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE,       \
                          false);                                              \
  }

#ifdef STOPTHEPOP_FASTBUILD
#define CALL_HIER_HEAD(HIER_CULLING, MID_QUEUE_SIZE)                           \
  switch (splatting_settings.sort_settings.queue_sizes.per_pixel) {            \
  case 4: {                                                                    \
    CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, 4);                                \
    break;                                                                     \
  }                                                                            \
  default: {                                                                   \
    throw std::runtime_error("Not supported head queue size");                 \
  }                                                                            \
  }

#define CALL_HIER_MID(HIER_CULLING)                                            \
  switch (splatting_settings.sort_settings.queue_sizes.tile_2x2) {             \
  case 8: {                                                                    \
    CALL_HIER_HEAD(HIER_CULLING, 8);                                           \
    break;                                                                     \
  }                                                                            \
  default: {                                                                   \
    throw std::runtime_error("Not supported mid queue size");                  \
  }                                                                            \
  }
#else // STOPTHEPOP_FASTBUILD
#define CALL_HIER_HEAD(HIER_CULLING, MID_QUEUE_SIZE)                           \
  switch (splatting_settings.sort_settings.queue_sizes.per_pixel) {            \
  case 4: {                                                                    \
    CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, 4);                                \
    break;                                                                     \
  }                                                                            \
  case 8: {                                                                    \
    CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, 8);                                \
    break;                                                                     \
  }                                                                            \
  case 16: {                                                                   \
    CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, 16);                               \
    break;                                                                     \
  }                                                                            \
  default: {                                                                   \
    throw std::runtime_error("Not supported head queue size");                 \
  }                                                                            \
  }

#define CALL_HIER_MID(HIER_CULLING)                                            \
  switch (splatting_settings.sort_settings.queue_sizes.tile_2x2) {             \
  case 8: {                                                                    \
    CALL_HIER_HEAD(HIER_CULLING, 8);                                           \
    break;                                                                     \
  }                                                                            \
  case 12: {                                                                   \
    CALL_HIER_HEAD(HIER_CULLING, 12);                                          \
    break;                                                                     \
  }                                                                            \
  case 20: {                                                                   \
    CALL_HIER_HEAD(HIER_CULLING, 20);                                          \
    break;                                                                     \
  }                                                                            \
  default: {                                                                   \
    throw std::runtime_error("Not supported mid queue size");                  \
  }                                                                            \
  }
#endif // STOPTHEPOP_FASTBUILD

      if (splatting_settings.culling_settings.hierarchical_4x4_culling) {
        CALL_HIER_MID(true);
      } else {
        CALL_HIER_MID(false);
      }

#undef CALL_HIER_MID
#undef CALL_HIER_HEAD
#undef CALL_HIER
#undef CALL_HIER_EXACT_DEPTH
#undef CALL_HIER_DEBUG
    }
  };
};

/*
extern template class ExtraFeaturesBlender<1>;
extern template class ExtraFeaturesBlender<2>;
extern template class ExtraFeaturesBlender<3>;
extern template class ExtraFeaturesBlender<4>;
extern template class ExtraFeaturesBlender<6>;
extern template class ExtraFeaturesBlender<8>;
extern template class ExtraFeaturesBlender<12>;
extern template class ExtraFeaturesBlender<16>;
*/
#endif