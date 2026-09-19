#include "xil_printf.h"
#include "xil_io.h"
#include "xsgm_kernel.h"
#include "xparameters.h"
#include "xstatus.h"
#include <stdint.h>
#include <math.h>
#include "sgm_config.hpp"
#include "left_img.hpp"
#include "right_img.hpp"
#include "gt_disp.hpp"
#include "xtime_l.h"

#define LEFT_BRAM_BASE 0x40010000U
#define RIGHT_BRAM_BASE 0x40018000U
#define DISP_BRAM_BASE 0x40020000U

int main ()
{
	XSgm_kernel sgm;

	int status = XSgm_kernel_Initialize(&sgm,
			XPAR_SGM_KERNEL_0_DEVICE_ID);

	if(status != XST_SUCCESS)
	{
		xil_printf("SGM driver init failed\r\n");
		while(1);
	}
	xil_printf("SGM driver init PASS\r\n");

	XTime t_total_start, t_total_end;
	XTime t_input_end;
	XTime t_pl_start, t_pl_end;
	XTime t_output_end;

	XTime_GetTime(&t_total_start);

	for(int r = 0; r < IMG_H; ++r)
	{
		for(int c = 0; c < IMG_W; c += 4)
		{
			uint32_t left_word = ((uint32_t)left_img[r][c + 0])
					| ((uint32_t)left_img[r][c + 1] << 8)
					| ((uint32_t)left_img[r][c + 2] << 16)
					| ((uint32_t)left_img[r][c + 3] << 24);

			uint32_t right_word = ((uint32_t)right_img[r][c + 0])
								| ((uint32_t)right_img[r][c + 1] << 8)
								| ((uint32_t)right_img[r][c + 2] << 16)
								| ((uint32_t)right_img[r][c + 3] << 24);

			int word_idx = (r* IMG_W + c) >> 2;

			Xil_Out32(LEFT_BRAM_BASE + word_idx * 4, left_word);
			Xil_Out32(RIGHT_BRAM_BASE + word_idx * 4, right_word);
		}
	}
	XTime_GetTime(&t_input_end);

	/* start kernel */
	XTime_GetTime(&t_pl_start);
	XSgm_kernel_Start(&sgm);

	while(!XSgm_kernel_IsDone(&sgm));

	XTime_GetTime(&t_pl_end);

	static uint8_t disp_img[IMG_H][IMG_W];

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
	XTime_GetTime(&t_output_end);
	XTime_GetTime(&t_total_end);

	xil_printf("SGM run completed \r\n");

	double total_time_ms = 1000.0 * (double)(t_total_end - t_total_start)
			/ (double)COUNTS_PER_SECOND;

	double input_ms = 1000.0 * (double)(t_input_end - t_total_start)
			/ (double)COUNTS_PER_SECOND;

	double pl_ms = 1000.0 * (double)(t_pl_end - t_pl_start)
			/ (double)COUNTS_PER_SECOND;

	double output_ms = 1000.0 * (double)(t_output_end - t_pl_end)
			/ (double)COUNTS_PER_SECOND;

	int fps_x100 = (int)((1000.0 / total_time_ms) * 100.0);

	int total_us  = (int)(total_time_ms * 1000.0);
	int input_us  = (int)(input_ms * 1000.0);
	int pl_us     = (int)(pl_ms * 1000.0);
	int output_us = (int)(output_ms * 1000.0);

	xil_printf("Total frame time: %d.%03d ms\r\n", total_us / 1000,
			total_us % 1000);
	xil_printf("Frames input time: %d.%03d ms\r\n", input_us / 1000,
			input_us % 1000);
	xil_printf("PL accelerator time: %d.%03d ms\r\n", pl_us / 1000,
			pl_us % 1000);
	xil_printf("Disparity output time: %d.%03d ms\r\n", output_us / 1000,
			output_us % 1000);

	xil_printf("Throughput: %d.%02d FPS\r\n", fps_x100 / 100, fps_x100 % 100);


	xil_printf("disp(48,160) = %d\r\n", (int)disp_img[48][160]);

	xil_printf("disp(48,161) = %d\r\n", (int)disp_img[48][161]);

	xil_printf("disp(48,162) = %d\r\n", (int)disp_img[48][162]);

	xil_printf("disp(48,163) = %d\r\n", (int)disp_img[48][163]);

	/* Accuracy evaluation */
	int eval_valid_count = 0;
	int eval_invalid_count = 0;

	int bad1 = 0;
	int bad3 = 0;

	double sum_abs_err = 0.0;

	const int cx = WIN >> 1;

	const int valid_r_min = WIN - 1;
	const int valid_c_min = (DISP - 1) + cx;
	const int valid_c_max = IMG_W - cx;

	for (int r = 0; r < IMG_H; ++r)
	{
	    for (int c = 0; c < IMG_W; ++c)
	    {
	        float gt = gt_disp[r][c];

	        bool gt_valid = (gt > 0.0f);

	        bool roi_valid =
	            (r >= valid_r_min) &&
	            (c >= valid_c_min) &&
	            (c < valid_c_max);

	        bool disp_range_valid =
	            (gt >= 0.0f) &&
	            (gt < DISP);

	        bool eval_valid =
	            gt_valid &&
	            roi_valid &&
	            disp_range_valid;

	        if (!eval_valid)
	        {
	            ++eval_invalid_count;
	            continue;
	        }

	        ++eval_valid_count;

	        float est = (float)disp_img[r][c];

	        float err = fabsf(est - gt);

	        sum_abs_err += err;

	        if (err > 1.0f)
	            ++bad1;

	        if (err > 3.0f)
	            ++bad3;
	    }
	}

	if (eval_valid_count == 0)
	{
	    xil_printf("ERROR: No valid GT pixels\r\n");
	    while (1);
	}

	double mae = sum_abs_err / (double)eval_valid_count;

	double bad1_pct = 100.0 * (double)bad1 / (double)eval_valid_count;

	double bad3_pct = 100.0 * (double)bad3 / (double)eval_valid_count;

	int mae_x1000  = (int)(mae * 1000.0);
	int bad1_x100 = (int)(bad1_pct * 100.0);
	int bad3_x100 = (int)(bad3_pct * 100.0);

	xil_printf("Evaluation-valid pixels: %d\r\n", eval_valid_count);

	xil_printf("Excluded pixels: %d\r\n", eval_invalid_count);

	xil_printf("MAE: %d.%03d px\r\n", mae_x1000 / 1000, mae_x1000 % 1000);

	xil_printf("Bad >1 px: %d.%02d%%\r\n", bad1_x100 / 100, bad1_x100 % 100);

	xil_printf("Bad >3 px: %d.%02d%%\r\n", bad3_x100 / 100, bad3_x100 % 100);

	while(1);

	return 0;
}
