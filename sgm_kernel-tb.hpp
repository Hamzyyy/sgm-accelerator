#ifndef SGM_KERNEL_TB_HPP
#define SGM_KERNEL_TB_HPP

#include <opencv2/opencv.hpp>
#include "hls_stream.h"
#include "sgm_params.hpp"

void sgm_kernel(hls::stream<pix_t>& left,
                hls::stream<pix_t>& right,
                hls::stream<disp_t>& disp);

#endif // SGM_KERNEL_TB_HPP

