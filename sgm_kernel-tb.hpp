#ifndef SGM_KERNEL_TB_HPP
#define SGM_KERNEL_TB_HPP

#include <opencv2/opencv.hpp>
#include "hls_stream.h"
#include "sgm_params.hpp"
#include "sgm_config.hpp"

void sgm_kernel(pix_t left[IMG_H][IMG_W],
                pix_t right[IMG_H][IMG_W],
                disp_t disp[IMG_H][IMG_W]);

#endif // SGM_KERNEL_TB_HPP
