#include "sgm_config.hpp"
#include "left_img.hpp"
#include "right_img.hpp"
#include "xil_printf.h"
#include "xtime_l.h"
#include "xsgm_kernel.h"
#include "xparameters.h"
#include "xil_cache.h"
#include "xstatus.h"

int main()
{
	xil_printf("Starting PL-PS Variant-B... \r\n");
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

	Xil_DCacheFlushRange((UINTPTR)&left_buf[0][0], IMG_H * IMG_W * sizeof(uint8_t));
	Xil_DCacheFlushRange((UINTPTR)&right_buf[0][0], IMG_H * IMG_W * sizeof(uint8_t));
	Xil_DCacheFlushRange((UINTPTR)&disp_hw[0][0], IMG_H * IMG_W * sizeof(uint8_t));

	XSgm_kernel_Start(&pl_accel);
	while(!XSgm_kernel_IsDone(&pl_accel));

	Xil_DCacheInvalidateRange((UINTPTR)&disp_hw[0][0], IMG_H * IMG_W * sizeof(uint8_t));

	XTime_GetTime(&t1);

	volatile uint32_t checksum = 0;
	for(int r = 0; r < IMG_H; ++r)
	{
		for(int c = 0; c < IMG_W; ++c)
		{
			checksum += disp_hw[r][c];
		}
	}

	uint64_t cycles = uint64_t(t1 - t0);
    uint32_t us = (uint32_t)((cycles * 1000000ULL) / COUNTS_PER_SECOND);

	xil_printf("SGM finished. \r\n");
	xil_printf("checksum = %u\r\n", (unsigned int)checksum);
	xil_printf("Timer counts = %u\r\n", (unsigned int)cycles);
    xil_printf("Variant B latency = %u us\r\n", (unsigned int)us);
    xil_printf("Variant B latency = %u ms\r\n", (unsigned int)(us / 1000));

	xil_printf("Counts per second = %d\r\n", COUNTS_PER_SECOND);

	xil_printf("disp(48, 80) = %d\r\n", disp_hw[48][80]);
	xil_printf("disp(48, 160) = %d\r\n", disp_hw[48][160]);
	xil_printf("disp(48, 240) = %d\r\n", disp_hw[48][240]);

	while(1);
}
