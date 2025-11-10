#pragma once
#include <ap_int.h>
#include <cstdint>
#include <limits>

/*---------------------------------------------------------------------------*/
using pix_t  = ap_uint<8>;
using disp_t = ap_uint<8>;
using cost_t = ap_uint<12>;
using wide_t = ap_uint<16>;

constexpr int IMG_H = 375;
constexpr int IMG_W = 1242;
constexpr int DISP  = 128;
constexpr int WIN   = 3;

static const cost_t INF_COST = cost_t(4095);

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

static inline cost_t min4(cost_t a, cost_t b, cost_t c, cost_t d)
{
#pragma HLS INLINE
  cost_t m1 = (a < b) ? a : b;
  cost_t m2 = (c < d) ? c : d;
  return (m1 < m2) ? m1 : m2;
}

static inline cost_t min_array(const cost_t *v, int n)
{
#pragma HLS INLINE
  cost_t m = INF_COST;
  for (int i = 0; i < n; ++i)
  {
  #pragma HLS UNROLL
    if (v[i] < m) m = v[i];
  }
  return m;
}

#include <hls_stream.h>
#include "common/xf_common.hpp"
#include "common/xf_video_mem.hpp"
