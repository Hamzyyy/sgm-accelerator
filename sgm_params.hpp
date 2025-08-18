#pragma once
#include <ap_int.h>
#include <cstdint>
#include <limits>

/*---------------------------------------------------------------------------*/
using pix_t  = ap_uint<8>;        // 8-bit grayscale pixel
using disp_t = ap_uint<8>;
using cost_t = ap_uint<12>;       // enough bits for aggregated cost
using wide_t = ap_uint<16>;

constexpr int IMG_H = 375;
constexpr int IMG_W = 1242;
constexpr int DISP  = 128;         // max disparity
constexpr int WIN   = 3;          // 3×3 spatial window

/*---------------------------------------------------------------------------*/
static inline pix_t absdiff(pix_t a, pix_t b)
{
#pragma HLS INLINE
    return (a > b) ? a - b : b - a;
}

static inline cost_t sat12(unsigned val)
{
#pragma HLS INLINE
    return (val > 4095u) ? cost_t(4095) : cost_t(val);
}

#include <hls_stream.h>
#include "common/xf_common.hpp"
#include "common/xf_video_mem.hpp"
