#include "sgm_config.hpp"
#include "sgm_sw_core.hpp"
#include "left_img.hpp"
#include "right_img.hpp"
#include "gt_disp.hpp"
#include "xil_printf.h"
#include "xtime_l.h"

int main()
{
	static uint8_t disp_out[IMG_H][IMG_W];

	xil_printf("Starting Variant A - PS SGM profiling... \r\n");

	sgm_sw_core(left_img, right_img, disp_out);

	XTime t0, t1;

	XTime_GetTime(&t0);
	sgm_sw_core(left_img, right_img, disp_out);
	XTime_GetTime(&t1);

	volatile uint32_t checksum = 0;
	for(int r = 0; r < IMG_H; ++r)
	{
		for(int c = 0; c < IMG_W; ++c)
		{
			checksum +=disp_out[r][c];
		}
	}

	/* Evaluation metrics */
	int total_pixels = 0;
	int eval_valid_count = 0, eval_invalid_count = 0;
	int zero_count_on_gt_valid= 0;
	int nonzero_count_on_gt_valid= 0;
	int out_of_range_on_gt_valid= 0;
	int bad1 = 0, bad3 = 0;
	double sum_abs_err = 0.0;

	const int cx = WIN >> 1;

	const int valid_r_min = WIN - 1;
	const int valid_c_min = (DISP - 1) + cx;
	const int valid_c_max = IMG_W - cx;

	for (int r = 0; r < IMG_H; ++r)
	{
		for (int c = 0; c < IMG_W; ++c)
		{
			++total_pixels;
			float gt_val = gt_disp[r][c];

	        bool gt_valid = gt_val > 0.0f;
	        bool roi_valid =
	        		(r >= valid_r_min) &&
	                (c >= valid_c_min) &&
	                (c < valid_c_max);

	        bool disp_range_valid =
	        		(gt_val >= 0.0f) &&
	                (gt_val < DISP);

	        bool eval_valid = gt_valid && roi_valid && disp_range_valid;

	    	if(!eval_valid)
	    	{
	    		++eval_invalid_count;
	    		continue;
	    	}
	    	++eval_valid_count;

	    	float est_disp = float(disp_out[r][c]);

	    	if(est_disp == 0.0f)
	    		++zero_count_on_gt_valid;
	    	else
	    		++nonzero_count_on_gt_valid;

	    	if(est_disp < 0.0f || est_disp >= DISP)
	    		++out_of_range_on_gt_valid;

	    	float err= est_disp - gt_val;
	    	if(err < 0) err = -err;

	    	sum_abs_err += err;
	    	if (err > 1.0f) bad1++;
	    	if (err > 3.0f) bad3++;
	    }
	}

	uint64_t cycles = uint64_t(t1 - t0);
    uint32_t us = (uint32_t)((cycles * 1000000ULL) / COUNTS_PER_SECOND);

	xil_printf("SGM finished. \r\n");
	xil_printf("checksum = %u\r\n", (unsigned int)checksum);
	xil_printf("Timer counts = %u\r\n", (unsigned int)cycles);
    xil_printf("Variant A-Census = %u us\r\n", (unsigned int)us);
    xil_printf("Variant A-Census = %u ms\r\n", (unsigned int)(us / 1000));

	xil_printf("Counts per second = %d\r\n", COUNTS_PER_SECOND);

	xil_printf("disp(48, 80) = %d\r\n", disp_out[48][80]);
	xil_printf("disp(48, 160) = %d\r\n", disp_out[48][160]);
	xil_printf("disp(48, 240) = %d\r\n", disp_out[48][240]);

	xil_printf("Line Buffer = %u ms\r\n", (unsigned int) ((t_linebuffer * 1000ULL) / COUNTS_PER_SECOND));
	xil_printf("Sliding Windows = %u ms\r\n", (unsigned int) ((t_slidingwindow * 1000ULL) / COUNTS_PER_SECOND));
	xil_printf("Census Cost = %u ms\r\n", (unsigned int) ((t_computecensus * 1000ULL) / COUNTS_PER_SECOND));
	xil_printf("Aggregate Cost = %u ms\r\n", (unsigned int) ((t_aggregatecost * 1000ULL) / COUNTS_PER_SECOND));
	xil_printf("Commit Prev. Costs = %u ms\r\n", (unsigned int) ((t_commitcosts * 1000ULL) / COUNTS_PER_SECOND));

    if (eval_valid_count == 0)
    {
        xil_printf("ERROR: No valid GT pixels for comparison\n");
        return 6;
    }

    unsigned int mae_x1000  = (unsigned int)((sum_abs_err * 1000.0) / eval_valid_count);
    unsigned int bad1_x100  = (unsigned int)((bad1 * 10000.0) / eval_valid_count);
    unsigned int bad3_x100  = (unsigned int)((bad3 * 10000.0) / eval_valid_count);

    xil_printf("Variant-A-Census MAE = %u.%03u px\r\n", mae_x1000 / 1000, mae_x1000 % 1000);
    xil_printf("Variant-A-Census Bad >1 px = %u.%02u %%\r\n", bad1_x100 / 100, bad1_x100 % 100);
    xil_printf("Variant-A-Census Bad >3 px = %u.%02u %%\r\n", bad3_x100 / 100, bad3_x100 % 100);

	while(1);
}
