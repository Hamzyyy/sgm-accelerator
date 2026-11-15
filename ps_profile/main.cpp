#include <stdint.h>
#include <cmath>
#include "xil_printf.h"
#include "xil_io.h"
#include "xsgm_kernel.h"
#include "xparameters.h"
#include "xstatus.h"
#include "xtime_l.h"
#include "sgm_config.hpp"
#include "dataset_frames.hpp"

#define CHECK_OUTPUT 0
#define PROFILE_STAGES 0

#define LEFT_BRAM_BASE 	XPAR_AXI_BRAM_CTRL_0_S_AXI_BASEADDR
#define RIGHT_BRAM_BASE XPAR_AXI_BRAM_CTRL_1_S_AXI_BASEADDR
#define DISP_BRAM_BASE 	XPAR_AXI_BRAM_CTRL_2_S_AXI_BASEADDR

int main ()
{
	/* Evaluation metrics */
	int kitti_valid_frame = 0;

	double total_kitti_d1_error = 0;
	double total_mae = 0;
	double total_Bad1_error = 0, total_Bad3_error = 0;

	uint64_t total_latency = 0;

#if PROFILE_STAGES
	uint64_t total_t_input = 0;
	uint64_t total_t_pl = 0;
	uint64_t total_t_output = 0;
#endif

	static double d1_per_frame[NUM_FRAMES];
	static double mae_per_frame[NUM_FRAMES];
	static double bad1_per_frame[NUM_FRAMES];
	static double bad3_per_frame[NUM_FRAMES];
	static uint32_t latency_per_frame[NUM_FRAMES];

	static uint8_t disp_img[IMG_H][IMG_W];

	xil_printf("Starting Variant B - PL-PS SGM evaluation... \r\n");

	XSgm_kernel sgm;

	int status = XSgm_kernel_Initialize(&sgm,
			XPAR_SGM_KERNEL_0_DEVICE_ID);

	if(status != XST_SUCCESS)
	{
		xil_printf("SGM driver init failed\r\n");
		while(1);
	}
	xil_printf("SGM driver init PASS\r\n");

	xil_printf("Running warm-up frame...\r\n");

	for(int r = 0; r < IMG_H; ++r)
	{
	    for(int c = 0; c < IMG_W; c += 4)
	    {
	        uint32_t left_word =
	              ((uint32_t)left_frames[0][r][c + 0])
	            | ((uint32_t)left_frames[0][r][c + 1] << 8)
	            | ((uint32_t)left_frames[0][r][c + 2] << 16)
	            | ((uint32_t)left_frames[0][r][c + 3] << 24);

	        uint32_t right_word =
	              ((uint32_t)right_frames[0][r][c + 0])
	            | ((uint32_t)right_frames[0][r][c + 1] << 8)
	            | ((uint32_t)right_frames[0][r][c + 2] << 16)
	            | ((uint32_t)right_frames[0][r][c + 3] << 24);

	        int word_idx = (r * IMG_W + c) >> 2;

	        Xil_Out32(LEFT_BRAM_BASE  + word_idx * 4, left_word);
	        Xil_Out32(RIGHT_BRAM_BASE + word_idx * 4, right_word);
	    }
	}

	/* Run accelerator */
	XSgm_kernel_Start(&sgm);
	while(!XSgm_kernel_IsDone(&sgm));

	for(int r = 0; r < IMG_H; ++r)
	{
	    for(int c = 0; c < IMG_W; c += 4)
	    {
	        int word_idx = (r * IMG_W + c) >> 2;

	        uint32_t word =
	            Xil_In32(DISP_BRAM_BASE + word_idx * 4);

	        disp_img[r][c + 0] = (uint8_t)( word        & 0xFF);
	        disp_img[r][c + 1] = (uint8_t)((word >>  8) & 0xFF);
	        disp_img[r][c + 2] = (uint8_t)((word >> 16) & 0xFF);
	        disp_img[r][c + 3] = (uint8_t)((word >> 24) & 0xFF);
	    }
	}

	xil_printf("Warm-up complete\r\n");

	for(int i = 0; i < NUM_FRAMES;++i)
	{
		if (i % 20 == 0)
			xil_printf("Frame %d\r\n", i);
		const float scale_x = scale_x_frames[i];

		XTime frame_start, frame_end;

#if PROFILE_STAGES
XTime t0, t1;
#endif

		XTime_GetTime(&frame_start);

#if PROFILE_STAGES
		XTime_GetTime(&t0);
#endif

		for(int r = 0; r < IMG_H; ++r)
		{
			for(int c = 0; c < IMG_W; c += 4)
			{
				uint32_t left_word = ((uint32_t)left_frames[i][r][c + 0])
						| ((uint32_t)left_frames[i][r][c + 1] << 8)
						| ((uint32_t)left_frames[i][r][c + 2] << 16)
						| ((uint32_t)left_frames[i][r][c + 3] << 24);

				uint32_t right_word = ((uint32_t)right_frames[i][r][c + 0])
									| ((uint32_t)right_frames[i][r][c + 1] << 8)
									| ((uint32_t)right_frames[i][r][c + 2] << 16)
									| ((uint32_t)right_frames[i][r][c + 3] << 24);

				int word_idx = (r* IMG_W + c) >> 2;

				Xil_Out32(LEFT_BRAM_BASE + word_idx * 4, left_word);
				Xil_Out32(RIGHT_BRAM_BASE + word_idx * 4, right_word);
			}
		}
#if PROFILE_STAGES
		XTime_GetTime(&t1);
		total_t_input += uint64_t (t1-t0);
#endif

	/* start kernel */
#if PROFILE_STAGES
	XTime_GetTime(&t0);
#endif

	XSgm_kernel_Start(&sgm);
	while(!XSgm_kernel_IsDone(&sgm));

#if PROFILE_STAGES
	XTime_GetTime(&t1);
	total_t_pl += uint64_t (t1-t0);
#endif


#if PROFILE_STAGES
	XTime_GetTime(&t0);
#endif
	for(int r = 0; r < IMG_H; ++r)
	{
		for(int c = 0; c < IMG_W; c += 4)
		{
			int word_idx = (r * IMG_W + c) >> 2;

			uint32_t word = Xil_In32(DISP_BRAM_BASE + word_idx * 4);

			disp_img[r][c + 0] = (uint8_t)( word        & 0xFF);
	        disp_img[r][c + 1] = (uint8_t)((word >>  8) & 0xFF);
	        disp_img[r][c + 2] = (uint8_t)((word >> 16) & 0xFF);
	        disp_img[r][c + 3] = (uint8_t)((word >> 24) & 0xFF);
		}
	}

#if PROFILE_STAGES
	XTime_GetTime(&t1);
	total_t_output += uint64_t (t1-t0);
#endif

	XTime_GetTime(&frame_end);
	uint64_t frame_ticks = (uint64_t)(frame_end - frame_start);

    uint32_t latency_us = (uint32_t)((frame_ticks * 1000000ULL) / COUNTS_PER_SECOND);
    latency_per_frame[i] = latency_us;

    if (i % 20 == 0)
    	xil_printf("Latency = %u us\r\n", (unsigned int)latency_us);

    total_latency += latency_us;

#if CHECK_OUTPUT
	uint32_t checksum = 0;

	for(int r = 0; r < IMG_H; ++r)
	{
	    for(int c = 0; c < IMG_W; ++c)
	    {
	        checksum += disp_img[r][c];
	    }
	}

	xil_printf("checksum = %u\r\n", (unsigned int)checksum);

#endif

	int kitti_d1_err = 0;
	int kitti_valid_count = 0;
	int kitti_invalid_count = 0;

	int bad1 = 0, bad3 = 0;
	double sum_abs_err = 0.0;

	for (int r = 0; r < IMG_H; ++r)
	{
		for (int c = 0; c < IMG_W; ++c)
		{
			float gt_disp = gt_frames[i][r][c];
			float est_disp = disp_img[r][c];

			bool gt_valid = gt_disp > 0.0f;
			if(!gt_valid)
			{
				// Invalid kitti ground truth
				kitti_invalid_count++;
				continue;
			}
			++kitti_valid_count;
			float err = (std::abs(est_disp - gt_disp)) / scale_x;

	    	sum_abs_err += err;
	    	if (err > 1.0f) bad1++;
	    	if (err > 3.0f) bad3++;

			/* Kitt-style D1 */
			bool abs_error_gt3 = err > 3.0f;
			bool rel_error_gt5 = ((err / (gt_disp/scale_x)) > 0.05f);

			if(abs_error_gt3 && rel_error_gt5)
			{
				++kitti_d1_err;
			}
	    }
	}
	double current_kitti_error = 0.0f;
	double current_mae = 0.0f;
	double current_bad1 = 0.0f;
	double current_bad3= 0.0f;

    if(kitti_valid_count > 0)
    {
    	current_kitti_error = (100.0 * kitti_d1_err / kitti_valid_count);
    	++kitti_valid_frame;
    	total_kitti_d1_error += current_kitti_error;

    	current_mae = (sum_abs_err / kitti_valid_count);
		current_bad1 = (100.0 * bad1 / kitti_valid_count);
		current_bad3= (100.0 * bad3 / kitti_valid_count);

		total_mae += current_mae;
		total_Bad1_error += current_bad1;
		total_Bad3_error +=current_bad3;
    }
    else
    {
        xil_printf( "ERROR: No KITTI-valid GT pixels at frame %d\n", i);
        return 5;
    }

	d1_per_frame[i] = current_kitti_error;
	mae_per_frame[i] = current_mae;
	bad1_per_frame[i] = current_bad1;
	bad3_per_frame[i] = current_bad3;

    unsigned int d1_per_frame_x1000  = (unsigned int)((d1_per_frame[i] * 1000.0));
    unsigned int mae_per_frame_x1000  = (unsigned int)((mae_per_frame[i] * 1000.0));
    unsigned int bad1_per_frame_x1000  = (unsigned int)((bad1_per_frame[i] * 1000.0));
    unsigned int bad3_per_frame_x1000  = (unsigned int)((bad3_per_frame[i] * 1000.0));

    if (i % 20 == 0)
    {
    	xil_printf ("Frame %d, ", i);
    	xil_printf (" D1 = %u.%03u", d1_per_frame_x1000/1000, d1_per_frame_x1000%1000);
    	xil_printf (" MAE = %u.%03u", mae_per_frame_x1000/1000, mae_per_frame_x1000%1000);
    	xil_printf (" Bad(>1) = %u.%03u", bad1_per_frame_x1000/1000, bad1_per_frame_x1000%1000);
    	xil_printf (" Bad(>3) = %u.%03u\n", bad3_per_frame_x1000/1000, bad3_per_frame_x1000%1000);
    }

    if((kitti_invalid_count + kitti_valid_count) != (IMG_H * IMG_W))
    {
    	xil_printf("ERROR: GT validity counts do not match image "
    			"size at frame: %d\n", i);
    }
}
////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////
		if(NUM_FRAMES > 0)
		{
			xil_printf("---Mean Latency--- \n ");

			double mean_latency = static_cast<double>(total_latency) / NUM_FRAMES;
			double squared_diff_sum = 0.0;

			for(int i = 0; i < NUM_FRAMES; ++i)
			{
				double diff = latency_per_frame[i] - mean_latency;
				squared_diff_sum += diff * diff;
			}
			double std_latency = std::sqrt(squared_diff_sum / NUM_FRAMES);

			uint32_t mean_latency_x1000  = (uint32_t)((mean_latency * 1000.0));
			uint32_t std_latency_x1000  = (uint32_t)((std_latency * 1000.0));
	        uint64_t fps_x1000 = ((uint64_t)NUM_FRAMES * 1000000000ULL) / total_latency;

	    	xil_printf("Mean latency = %u.%03u us\r\n",
	    			mean_latency_x1000/1000, mean_latency_x1000%1000);

	    	xil_printf("Latency Standard Deviation = %u.%03u us\r\n",
	    			std_latency_x1000/1000, std_latency_x1000%1000);

	    	xil_printf("Throughput = %u.%03u FPS\r\n",
	    			(unsigned int)(fps_x1000 / 1000), (unsigned int)(fps_x1000 % 1000));

	#if PROFILE_STAGES
	    	uint64_t total_profile_ticks = total_t_input + total_t_pl +
	    			total_t_output;

	    	double avg_input_ms = ((double) total_t_input / NUM_FRAMES)
	    			* 1000.0 / COUNTS_PER_SECOND;

	    	double avg_pl_ms = ((double) total_t_pl / NUM_FRAMES)
	    			* 1000.0 / COUNTS_PER_SECOND;

	    	double avg_output_ms = ((double) total_t_output / NUM_FRAMES)
	    			* 1000.0 / COUNTS_PER_SECOND;


	    	double pct_input = (100.0 * total_t_input) /
	    			total_profile_ticks;

	    	double pct_pl = (100.0 * total_t_pl) /
	    			total_profile_ticks;

	    	double pct_output = (100.0 * total_t_output) /
	    			total_profile_ticks;

	    	uint32_t pct_input_x100		= (uint32_t)(100.0 * pct_input);
	    	uint32_t pct_pl_x100 		= (uint32_t)(100.0 * pct_pl);
	    	uint32_t pct_output_x100	= (uint32_t)(100.0 * pct_output);


	    	uint32_t input_x1000    	= (uint32_t)(1000.0 * avg_input_ms);
	    	uint32_t pl_ms_x1000 		= (uint32_t)(1000.0 * avg_pl_ms);
	    	uint32_t output_ms_x1000	= (uint32_t)(1000.0 * avg_output_ms);


	    	xil_printf("Image input time:\r\n");
	    	xil_printf("Mean = %u.%03u ms/frame\r\n",
	    			input_x1000/1000, input_x1000%1000);
	    	xil_printf("Share = %u.%02u %%\r\n",
	    			pct_input_x100/100, pct_input_x100%100);

	    	xil_printf("\r\n\r\n PL execution time:\r\n");
	    	xil_printf("Mean = %u.%03u ms/frame\r\n",
	    			pl_ms_x1000/1000, pl_ms_x1000%1000);
	    	xil_printf("Share = %u.%02u %%\r\n",
	    			pct_pl_x100/100, pct_pl_x100%100);

	    	xil_printf("\r\n\r\n Image output time:\r\n");
	    	xil_printf("Mean = %u.%03u ms/frame\r\n",
	    			output_ms_x1000/1000, output_ms_x1000%1000);
	    	xil_printf("Share = %u.%02u %%\r\n",
	    			pct_output_x100/100, pct_output_x100%100);
	#endif
		}

		if(kitti_valid_frame > 0)
		{
			xil_printf("---KITTI-style Evaluation --- \n ");

			double mean_d1 = total_kitti_d1_error / kitti_valid_frame;
			double squared_diff_sum = 0.0;

			for(int i = 0; i < NUM_FRAMES; ++i)
			{
				double diff = d1_per_frame[i] - mean_d1;
				squared_diff_sum += diff * diff;
			}
			double std_d1 = std::sqrt(squared_diff_sum / kitti_valid_frame);

	        unsigned int mean_d1_x1000  = (unsigned int)((mean_d1 * 1000.0));
	        unsigned int std_d1_x1000  = (unsigned int)((std_d1 * 1000.0));

	    	xil_printf("Mean D1 = %u.%03u\r\n", mean_d1_x1000/1000, mean_d1_x1000%1000);
	    	xil_printf("D1 Standard Deviation = %u.%03u\r\n", std_d1_x1000/1000, std_d1_x1000%1000);
		}

		if(kitti_valid_frame > 0)
		{
			xil_printf("\n --- Accuracy Evaluation --- \n ");

			double mean_mae = total_mae / kitti_valid_frame;
			double mae_squared_diff_sum = 0.0;

			double mean_bad1 = total_Bad1_error / kitti_valid_frame;
			double bad1_squared_diff_sum = 0.0;

			double mean_bad3 = total_Bad3_error / kitti_valid_frame;
			double bad3_squared_diff_sum = 0.0;

			for(int i = 0; i < NUM_FRAMES; ++i)
			{
				double mae_diff = mae_per_frame[i] - mean_mae;
				mae_squared_diff_sum += mae_diff * mae_diff;


				double bad1_diff = bad1_per_frame[i] - mean_bad1;
				bad1_squared_diff_sum += bad1_diff * bad1_diff;


				double bad3_diff = bad3_per_frame[i] - mean_bad3;
				bad3_squared_diff_sum += bad3_diff * bad3_diff;
			}
			double std_mae = std::sqrt(mae_squared_diff_sum / kitti_valid_frame);
			double std_bad1 = std::sqrt(bad1_squared_diff_sum / kitti_valid_frame);
			double std_bad3 = std::sqrt(bad3_squared_diff_sum / kitti_valid_frame);

	        unsigned int mean_mae_x1000  = (unsigned int)((mean_mae * 1000.0));
	        unsigned int std_mae_x1000  = (unsigned int)((std_mae * 1000.0));

	        unsigned int mean_bad1_x1000  = (unsigned int)((mean_bad1 * 1000.0));
	        unsigned int std_bad1_x1000  = (unsigned int)((std_bad1 * 1000.0));

	        unsigned int mean_bad3_x1000  = (unsigned int)((mean_bad3 * 1000.0));
	        unsigned int std_bad3_x1000  = (unsigned int)((std_bad3 * 1000.0));

	    	xil_printf("Mean MAE = %u.%03u\r\n", mean_mae_x1000/1000, mean_mae_x1000%1000);
	    	xil_printf("MAE Standard Deviation = %u.%03u\r\n", std_mae_x1000/1000, std_mae_x1000%1000);

	    	xil_printf("Mean Bad(>1) = %u.%03u\r\n", mean_bad1_x1000/1000, mean_bad1_x1000%1000);
	    	xil_printf("Bad(>1) Standard Deviation = %u.%03u\r\n", std_bad1_x1000/1000, std_bad1_x1000%1000);

	    	xil_printf("Mean Bad(>3) = %u.%03u\r\n", mean_bad3_x1000/1000, mean_bad3_x1000%1000);
	    	xil_printf("Bad(>3) Standard Deviation = %u.%03u\r\n", std_bad3_x1000/1000, std_bad3_x1000%1000);
		}


	while(1);

	return 0;
}
