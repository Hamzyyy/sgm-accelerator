#include "sgm_config.hpp"
#include "sgm_sw_core.hpp"
#include "left_img.hpp"
#include "right_img.hpp"
#include "xil_printf.h"
#include "xtime_l.h"

int main()
{
	static uint8_t disp_out[IMG_H][IMG_W];

	xil_printf("Starting PS SGM profiling... \r\n");

	sgm_sw_core(left_img, right_img, disp_out);

	XTime t0, t1;

	XTime_GetTime(&t0);
	sgm_sw_core(left_img, right_img, disp_out);

	volatile uint32_t checksum = 0;
	for(int r = 0; r < IMG_H; ++r)
	{
		for(int c = 0; c < IMG_W; ++c)
		{
			checksum +=disp_out[r][c];
		}
	}

	XTime_GetTime(&t1);

	uint64_t cycles = uint64_t(t1 - t0);
    uint32_t us = (uint32_t)((cycles * 1000000ULL) / COUNTS_PER_SECOND);

	xil_printf("SGM finished. \r\n");
	xil_printf("checksum = %u\r\n", (unsigned int)checksum);
	xil_printf("Timer counts = %u\r\n", (unsigned int)cycles);
    xil_printf("PS latency = %u us\r\n", (unsigned int)us);
    xil_printf("PS latency = %u ms\r\n", (unsigned int)(us / 1000));

	xil_printf("Counts per second = %d\r\n", COUNTS_PER_SECOND);

	xil_printf("disp(48, 80) = %d\r\n", disp_out[48][80]);
	xil_printf("disp(48, 160) = %d\r\n", disp_out[48][160]);
	xil_printf("disp(48, 240) = %d\r\n", disp_out[48][240]);

	while(1);
}
