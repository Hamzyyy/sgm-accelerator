#include "sgm_sw_core.hpp"
#include "xtime_l.h"

#define PROFILE_STAGES 0

uint64_t t_linebuffer = 0;
uint64_t t_slidingwindow = 0;
uint64_t t_computecensus = 0;
uint64_t t_aggregatecost = 0;
uint64_t t_commitcosts = 0;

static inline uint16_t sat12_core(unsigned v)
{
    return v > 63u ? 63u : uint16_t(v);
}

static inline uint8_t absdiff_core(uint8_t a, uint8_t b)
{
    return a > b ? a - b : b - a;
}

static void update_line_buffers_core(
		uint8_t bufL[WIN][IMG_W],
		uint8_t bufR[WIN][IMG_W],
		int c,
		uint8_t pL,
		uint8_t pR)
{
    for (int i = 0; i < WIN - 1; ++i)
    {
        bufL[i][c] = bufL[i + 1][c];
        bufR[i][c] = bufR[i + 1][c];
    }
    bufL[WIN - 1][c] = pL;
    bufR[WIN - 1][c] = pR;
}

static void update_sliding_windows_core(
		uint8_t bufL[WIN][IMG_W],
		uint8_t bufR[WIN][IMG_W],
		int c,
		uint8_t leftWin[WIN][WIN],
		uint8_t rightStripe[WIN][RIGHT_STRIPE_W],
		int& right_wr)
{
	    for (int wy = 0; wy < WIN; ++wy)
	    {
	        for (int wx = 0; wx < WIN - 1; ++wx)
	        {
	            leftWin[wy][wx] = leftWin[wy][wx + 1];
	        }
	    }

		for (int wy = 0; wy < WIN; ++wy)
		{
			leftWin[wy][WIN - 1] = bufL[wy][c];
	    }

		right_wr++;
		if(right_wr == RIGHT_STRIPE_W)
			right_wr = 0;

		for (int wy = 0; wy < WIN; ++wy)
		{
			rightStripe[wy][right_wr] = bufR[wy][c];
		}
}

static void compute_census_cost_vector_core(
		uint8_t leftWin[WIN][WIN],
		uint8_t rightStripe[WIN][RIGHT_STRIPE_W],
		int right_wr,
		int out_c,
		uint16_t curCost[DISP])
{
	for (int d = 0; d < DISP; ++d)
	{
		if(d > out_c)
		{
	        curCost[d] = INF_COST_core;
		}
		else
		{
			uint16_t sum = 0;

	    	int logicalIndex_center = RIGHT_STRIPE_W - WIN - d + CENSUS_CX;
	    	int physIndex_center= right_wr + 1 + logicalIndex_center;

	    	if (physIndex_center >= RIGHT_STRIPE_W)
	    		physIndex_center -= RIGHT_STRIPE_W;

	    	uint8_t centerL = leftWin[CENSUS_CY][CENSUS_CX];
	    	uint8_t centerR = rightStripe[CENSUS_CY][physIndex_center];

	        for (int wy = 0; wy < WIN; ++wy)
	        {
	            for (int wx = 0; wx < WIN; ++wx)
	            {
	            	if (wy == CENSUS_CY && wx == CENSUS_CX)
	            	{
	            	    continue;
	            	}
	            	else
	            	{
						int logicalIndex = RIGHT_STRIPE_W - WIN - d + wx;

						int physIndex = right_wr + 1 + logicalIndex;
						if (physIndex >= RIGHT_STRIPE_W)
								physIndex -= RIGHT_STRIPE_W;

						uint8_t lpx = leftWin[wy][wx];
						uint8_t rpx = rightStripe[wy][physIndex];

						bool bitL = (lpx < centerL);
						bool bitR = (rpx < centerR);

						sum += (bitL ^ bitR);
	            	}
	            }
	        }
	        curCost[d] = sum;
		}
	}
}

static uint8_t aggregate_paths_and_select_core(
    const uint16_t curCost[DISP],
    const uint16_t prevCostL[DISP],
    const uint16_t prevCostT_col[DISP],
	uint16_t minPrevLR,
	uint16_t minPrevTB,
	uint16_t aggLR_arr[DISP],
	uint16_t aggTB_arr[DISP],
	uint16_t aggCost[DISP],
	uint16_t& newMinLR,
	uint16_t& newMinTB)
{
	uint16_t bestCost = INF_COST_core;
    uint8_t bestDisp = 0;

    uint16_t runMinLR = INF_COST_core;
    uint16_t runMinTB = INF_COST_core;

    for (int d = 0; d < DISP; ++d)
    {
    	uint16_t p0_LR = prevCostL[d];
    	uint16_t p1_LR = (d > 0) ? sat12_core(prevCostL[d - 1] + P1_core) : INF_COST_core;
    	uint16_t p2_LR = (d < DISP - 1) ? sat12_core(prevCostL[d + 1] + P1_core) : INF_COST_core;
    	uint16_t p3_LR = sat12_core(minPrevLR + P2_core);

    	uint16_t minLR = p0_LR;
        if (p1_LR < minLR) minLR = p1_LR;
        if (p2_LR < minLR) minLR = p2_LR;
        if (p3_LR < minLR) minLR = p3_LR;

        uint16_t aggLR;
        if(curCost[d] == INF_COST_core)
        {
        	aggLR = INF_COST_core;
        }
        else
        {
        	aggLR = sat12_core(curCost[d] + minLR - minPrevLR);
        }
        aggLR_arr[d] = aggLR;

        uint16_t p0_TB = prevCostT_col[d];
        uint16_t p1_TB = (d > 0) ? sat12_core(prevCostT_col[d - 1] + P1_core) : INF_COST_core;
        uint16_t p2_TB = (d < DISP - 1) ? sat12_core(prevCostT_col[d + 1] + P1_core) : INF_COST_core;
        uint16_t p3_TB = sat12_core(minPrevTB + P2_core);

        uint16_t minTB = p0_TB;
        if (p1_TB < minTB) minTB = p1_TB;
        if (p2_TB < minTB) minTB = p2_TB;
        if (p3_TB < minTB) minTB = p3_TB;

        uint16_t aggTB;
        if(curCost[d] == INF_COST_core)
        {
        	aggTB = INF_COST_core;
        }
        else
        {
        	aggTB = sat12_core(curCost[d] + minTB - minPrevTB);
        }

        aggTB_arr[d] = aggTB;

        if(aggLR < runMinLR) runMinLR = aggLR;
        if(aggTB < runMinTB) runMinTB = aggTB;

        uint16_t sum2 = sat12_core(aggLR + aggTB);
        aggCost[d] = sum2;

        if (sum2 < bestCost)
        {
            bestCost = sum2;
            bestDisp = uint8_t(d);
        }
    }
    newMinLR = runMinLR;
    newMinTB = runMinTB;

    return bestDisp;
}

static void commit_prev_costs_core(
		uint16_t prevCostL[DISP],
		uint16_t prevCostT_col[DISP],
		const uint16_t aggLR_arr[DISP],
		const uint16_t aggTB_arr[DISP])
{
    for (int d = 0; d < DISP; ++d)
    {
        prevCostL[d]    = aggLR_arr[d];
        prevCostT_col[d] = aggTB_arr[d];
    }
}

void sgm_sw_core(
		const uint8_t left[IMG_H][IMG_W],
		const uint8_t right[IMG_H][IMG_W],
		uint8_t disp[IMG_H][IMG_W])
{
	t_linebuffer = 0;
	t_slidingwindow = 0;
	t_computecensus = 0;
	t_aggregatecost = 0;
	t_commitcosts = 0;

	XTime t0, t1;

	for (int r = 0; r < IMG_H; ++r)
	    for (int c = 0; c < IMG_W; ++c)
	        disp[r][c] = 0;

    static uint8_t bufL[WIN][IMG_W];
    static uint8_t bufR[WIN][IMG_W];

    for (int wy = 0; wy < WIN; ++wy)
    {
    	for (int c = 0; c < IMG_W; ++c)
    	{
    		bufL[wy][c] = 0;
            bufR[wy][c] = 0;
    	}
    }

    /* Cost arrays */
    static uint16_t curCost[DISP];
    static uint16_t prevCostL[DISP];
    static uint16_t prevCostT[IMG_W][DISP];
    static uint16_t aggCost[DISP];
    static uint16_t aggLR_arr[DISP];
    static uint16_t aggTB_arr[DISP];
    static uint8_t leftWin[WIN][WIN];
    static uint8_t rightStripe[WIN][RIGHT_STRIPE_W];
    static uint16_t minPrevT[IMG_W];

    /* center offset */
    const int cx = WIN >> 1;
    const int cy = WIN >> 1;


    for (int r = 0; r < IMG_H; r++)
    {
    	int right_wr = RIGHT_STRIPE_W - 1;
    	uint16_t minPrevLR = 0;
	    int out_r = r - cy;

        /* Reset aggregation for new row */
    	for (int d = 0; d < DISP; d++)
    	{
    		prevCostL[d] = uint16_t(0);
        }
    	if(r == 0)
    	{
    		for (int c = 0; c < IMG_W; ++c)
    		{
    			minPrevT[c] = 0;
    			for (int d = 0; d < DISP; ++d)
    			{
    				prevCostT[c][d] = uint16_t(0);
    			}
    		}
    	}

    	for (int wy = 0; wy < WIN; ++wy)
    	{
    		for (int wx = 0; wx < WIN; ++wx)
    		{
    			leftWin[wy][wx] = 0;
    		}
    	}

    	for (int wy = 0; wy < WIN; ++wy)
    	{
    		for (int k = 0; k < RIGHT_STRIPE_W; ++k)
    		{
    			rightStripe[wy][k] = 0;
    		}
    	}
    	for (int c = 0; c < IMG_W; ++c)
    	{
    		uint8_t pL = left[r][c];
    		uint8_t pR = right[r][c];
    	    int out_c = c - cx;

#if PROFILE_STAGES

    		XTime_GetTime(&t0);
#endif
    		update_line_buffers_core(bufL, bufR, c, pL, pR);
#if PROFILE_STAGES
    		XTime_GetTime(&t1);

    		t_linebuffer += (t1 - t0);
#endif

#if PROFILE_STAGES
    		XTime_GetTime(&t0);
#endif
    		update_sliding_windows_core(bufL, bufR, c, leftWin, rightStripe,
    				right_wr);
#if PROFILE_STAGES
    		XTime_GetTime(&t1);

    		t_slidingwindow += (t1 - t0);
#endif
    		const bool valid_window = (r >= WIN - 1) && (c < IMG_W);

    	    if (valid_window)
    	    {
#if PROFILE_STAGES
    	    	XTime_GetTime(&t0);
#endif
    	    	compute_census_cost_vector_core(leftWin, rightStripe, right_wr,
    	    			out_c, curCost);
#if PROFILE_STAGES
    	    	XTime_GetTime(&t1);

    	    	t_computecensus += (t1 - t0);
#endif
    	    }
    	    		if(out_r >= 0 && out_c >= 0)
    	    		{
    	    			uint8_t outDisp = 0;
    	    			if(valid_window)
    	    			{
        	    			uint16_t newMinLR = INF_COST_core;
        	    			uint16_t newMinTB = INF_COST_core;

#if PROFILE_STAGES
        	    			XTime_GetTime(&t0);
#endif
        	    			uint8_t bestDisp = aggregate_paths_and_select_core(curCost,
        	    					prevCostL, prevCostT[out_c], minPrevLR,
									minPrevT[out_c], aggLR_arr, aggTB_arr,
									aggCost, newMinLR, newMinTB);
#if PROFILE_STAGES
        	    			XTime_GetTime(&t1);

        	    			t_aggregatecost += (t1 - t0);
#endif

#if PROFILE_STAGES
        	    			XTime_GetTime(&t0);
#endif
        	    	        commit_prev_costs_core(prevCostL, prevCostT[out_c], aggLR_arr,
        	    	        		aggTB_arr);
#if PROFILE_STAGES
        	    	        XTime_GetTime(&t1);

        	    	        t_commitcosts += (t1 - t0);
#endif

        	    	        minPrevLR = newMinLR;
        	    	        minPrevT[out_c] = newMinTB;

        	    	        outDisp = bestDisp;
    	    			}
    	    	        disp[out_r][out_c] = outDisp;
    	    		}
    	    	}
    	if(out_r >=0)
    	{
	        for (int t = 0; t < cx; ++t)
	        {
	        	disp[out_r][IMG_W - cx + t] = 0;
	        }
    	}
    }
}
