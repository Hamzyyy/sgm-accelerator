#pragma once
#include <ap_int.h>
#include <cstdint>
#include <limits>
#include "sgm_config.hpp"
/*---------------------------------------------------------------------------*/
using pix_t  = ap_uint<8>;
using disp_t = ap_uint<8>;
using census_t = ap_uint<8>;
using cost_t = ap_uint<6>;
using wide_t = ap_uint<8>;
static const int COST_BITS = 6;
using group_t = ap_uint<PAR * COST_BITS>;

/* Penalties */
const cost_t P1 = cost_t(3);
const cost_t P2 = cost_t(15);


typedef ap_uint<32> bram_word_t;
static const int PIXEL_PER_WORD = 4;
static const int FRAME_PIXELS = IMG_H * IMG_W;
static const int FRAME_WORDS =(FRAME_PIXELS + PIXEL_PER_WORD - 1) / PIXEL_PER_WORD;

/*---------------------------------------------------------------------------*/
static inline pix_t absdiff(pix_t a, pix_t b)
{
#pragma HLS INLINE
    return (a > b) ? a - b : b - a;
}

static inline cost_t sat12(wide_t val)
{
#pragma HLS INLINE
    return (val > 63u) ? cost_t(63) : cost_t(val);
}

#include <hls_stream.h>
#include "common/xf_common.hpp"
#include "common/xf_video_mem.hpp"
