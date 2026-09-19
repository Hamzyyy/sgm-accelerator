#ifndef SGM_KERNEL_TB_HPP
#define SGM_KERNEL_TB_HPP

#include <opencv2/opencv.hpp>
#include "hls_stream.h"
#include "sgm_params.hpp"
#include "sgm_config.hpp"

void sgm_kernel(bram_word_t left[FRAME_WORDS],
				bram_word_t right[FRAME_WORDS],
				bram_word_t disp[FRAME_WORDS]);

#endif // SGM_KERNEL_TB_HPP
