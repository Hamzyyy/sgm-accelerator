#include "sgm_config.hpp"
#include "left_img.hpp"
#include "right_img.hpp"
#include "gt_disp.hpp"
#include "xil_printf.h"
#include "xtime_l.h"
#include "xsgm_kernel.h"
#include "xparameters.h"
#include "xil_cache.h"
#include "xstatus.h"

int main()
{
	xil_printf("Starting PL-PS Variant-B Census... \r\n");
	XTime t0, t1;

	XTime_GetTime(&t0);
	static uint8_t left_buf[IMG_H][IMG_W] __attribute__((aligned(32)));
	static uint8_t right_buf[IMG_H][IMG_W] __attribute__((aligned(32)));
	static uint8_t disp_hw[IMG_H][IMG_W] __attribute__((aligned(32)));

	for(int r = 0; r < IMG_H; ++r)
	{
		for(int c = 0; c < IMG_W; ++c)
		{
			left_buf[r][c] = left_img[r][c];
			right_buf[r][c] = right_img[r][c];
			disp_hw[r][c] = 0;
		}
	}
	XTime_GetTime(&t1);

	XTime t_input = t1 - t0;

	XSgm_kernel pl_accel;
	int status = XSgm_kernel_Initialize(&pl_accel,XPAR_SGM_KERNEL_0_DEVICE_ID);
	if(status != XST_SUCCESS)
	{
		xil_printf("SGM init failed...\r\n");
		return -1;
	}

	XSgm_kernel_Set_left_r(&pl_accel, (UINTPTR)&left_buf[0][0]);
	XSgm_kernel_Set_right_r(&pl_accel, (UINTPTR)&right_buf[0][0]);
	XSgm_kernel_Set_disp(&pl_accel, (UINTPTR)&disp_hw[0][0]);

	XTime_GetTime(&t0);
	Xil_DCacheFlushRange((UINTPTR)&left_buf[0][0], IMG_H * IMG_W * sizeof(uint8_t));
	Xil_DCacheFlushRange((UINTPTR)&right_buf[0][0], IMG_H * IMG_W * sizeof(uint8_t));
	Xil_DCacheFlushRange((UINTPTR)&disp_hw[0][0], IMG_H * IMG_W * sizeof(uint8_t));
	XTime_GetTime(&t1);

	XTime t_cache_flush = t1 - t0;

	XTime_GetTime(&t0);
	XSgm_kernel_Start(&pl_accel);
	while(!XSgm_kernel_IsDone(&pl_accel));
	XTime_GetTime(&t1);

	XTime t_pl= t1 - t0;

	XTime_GetTime(&t0);
	Xil_DCacheInvalidateRange((UINTPTR)&disp_hw[0][0], IMG_H * IMG_W * sizeof(uint8_t));
	XTime_GetTime(&t1);

	XTime t_cache_invalidate = t1 - t0;

	XTime_GetTime(&t0);
	volatile uint32_t checksum = 0;
	for(int r = 0; r < IMG_H; ++r)
	{
		for(int c = 0; c < IMG_W; ++c)
		{
			checksum += disp_hw[r][c];
		}
	}
	XTime_GetTime(&t1);
	XTime t_checksum = t1 - t0;

	XTime t_variant_b = t_input + t_cache_flush + t_pl + t_cache_invalidate;



	/* Evaluation metrics */
    int eval_valid_count = 0;
    int bad1_b = 0, bad3_b = 0;
    double sum_abs_err_b = 0.0;

    const int cx = WIN >> 1;

    const int valid_r_min = WIN - 1;
    const int valid_c_min = (DISP - 1) + cx;
    const int valid_c_max = IMG_W - cx;

    for (int r = 0; r < IMG_H; ++r)
    {
    	for (int c = 0; c < IMG_W; ++c)
    	{
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

            if (!eval_valid)
            {
                continue;
            }

            ++eval_valid_count;

    		float est_disp_b = float(disp_hw[r][c]);

    		float err_b= est_disp_b - gt_val;
    		if(err_b < 0) err_b = -err_b;

    		sum_abs_err_b += err_b;
    		if (err_b > 1.0f) bad1_b++;
    		if (err_b > 3.0f) bad3_b++;
    	}
    }

	xil_printf("SGM finished. \r\n");
	xil_printf("checksum = %u\r\n", (unsigned int)checksum);

    xil_printf("data input time = %u us \r\n", (unsigned)((t_input * 1000000ULL) / COUNTS_PER_SECOND));
    xil_printf("cache flush time = %u us \r\n", (unsigned)((t_cache_flush * 1000000ULL) / COUNTS_PER_SECOND));
    xil_printf("PL accelerator time = %u us \r\n", (unsigned)((t_pl * 1000000ULL) / COUNTS_PER_SECOND));
    xil_printf("cache invalidate time = %u us \r\n", (unsigned)((t_cache_invalidate * 1000000ULL) / COUNTS_PER_SECOND));
    xil_printf("checksum time = %u us \r\n", (unsigned)((t_checksum * 1000000ULL) / COUNTS_PER_SECOND));

    xil_printf("Variant B Census latency = %u us\r\n", (unsigned int)((t_variant_b * 1000000ULL) / COUNTS_PER_SECOND));
    xil_printf("Variant B Census latency = %u ms\r\n", (unsigned int)((t_variant_b * 1000ULL) / COUNTS_PER_SECOND));

	xil_printf("Counts per second = %d\r\n", COUNTS_PER_SECOND);

	xil_printf("disp(48, 80) = %d\r\n", disp_hw[48][80]);
	xil_printf("disp(48, 160) = %d\r\n", disp_hw[48][160]);
	xil_printf("disp(48, 240) = %d\r\n", disp_hw[48][240]);

    if (eval_valid_count == 0)
    {
        xil_printf("ERROR: No valid GT pixels for comparison\n");
        return 6;
    }

    unsigned int mae_x1000_b  = (unsigned int)((sum_abs_err_b * 1000.0) / eval_valid_count);
    unsigned int bad1_x100_b  = (unsigned int)((bad1_b * 10000.0) / eval_valid_count);
    unsigned int bad3_x100_b  = (unsigned int)((bad3_b * 10000.0) / eval_valid_count);

    xil_printf("Evaluation valid pixels = %u\r\n", eval_valid_count);
    xil_printf("Variant-B MAE = %u.%03u px\r\n", mae_x1000_b / 1000, mae_x1000_b % 1000);
    xil_printf("Variant-B Bad >1 px = %u.%02u %%\r\n", bad1_x100_b / 100, bad1_x100_b % 100);
    xil_printf("Variant-B Bad >3 px = %u.%02u %%\r\n", bad3_x100_b / 100, bad3_x100_b % 100);

	while(1);
}
