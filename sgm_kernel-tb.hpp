#ifndef SGM_KERNEL_TB_HPP
#define SGM_KERNEL_TB_HPP

#include <opencv2/opencv.hpp>
#include "hls_stream.h"
#include "sgm_params.hpp"
#include "sgm_config.hpp"

void sgm_kernel(pix_t left[IMG_H][IMG_W],
                pix_t right[IMG_H][IMG_W],
                disp_t disp[IMG_H][IMG_W]);

void sgm_sw(const cv::Mat& left,
		const cv::Mat& right,
		cv::Mat& disp);


#endif // SGM_KERNEL_TB_HPP

