#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "sgm_params.hpp"

static const uint16_t P1_SW = 10;
static const uint16_t P2_SW = 150;
static const uint16_t INF_COST_SW = 4095;

static inline uint16_t sat12_sw(unsigned v)
{
    return v > 4095u ? 4095u : uint16_t(v);
}

static inline uint8_t absdiff_sw(uint8_t a, uint8_t b)
{
    return a > b ? a - b : b - a;
}

void update_line_buffers_sw(
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

void update_sliding_windows_sw(
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

void compute_sad_cost_vector_sw(
		uint8_t leftWin[WIN][WIN],
		uint8_t rightStripe[WIN][RIGHT_STRIPE_W],
		int right_wr,
		uint16_t curCost[DISP])
{
	for(int d= 0; d < DISP; ++d)
	{
		uint16_t sum = 0;

		for(int wy = 0; wy < WIN; ++wy)
		{
			for(int wx = 0; wx < WIN; ++wx)
			{
				int logicalIndex = RIGHT_STRIPE_W - WIN - d + wx;
				int physIndex = right_wr + 1 + logicalIndex;

				if(physIndex >= RIGHT_STRIPE_W)
					physIndex -= RIGHT_STRIPE_W;

				uint8_t lpx = leftWin[wy][wx];
				uint8_t rpx = rightStripe[wy][physIndex];

				sum +=absdiff_sw(lpx, rpx);
			}
		}
		curCost[d] = sum;
	}
}

uint8_t aggregate_paths_and_select_sw(
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
	uint16_t bestCost = INF_COST_SW;
    uint8_t bestDisp = 0;

    uint16_t runMinLR = INF_COST_SW;
    uint16_t runMinTB = INF_COST_SW;

    for (int d = 0; d < DISP; ++d)
    {
    	uint16_t p0_LR = prevCostL[d];
    	uint16_t p1_LR = (d > 0) ? sat12_sw(prevCostL[d - 1] + P1_SW) : INF_COST_SW;
    	uint16_t p2_LR = (d < DISP - 1) ? sat12_sw(prevCostL[d + 1] + P1_SW) : INF_COST_SW;
    	uint16_t p3_LR = sat12_sw(minPrevLR + P2_SW);

    	uint16_t minLR = p0_LR;
        if (p1_LR < minLR) minLR = p1_LR;
        if (p2_LR < minLR) minLR = p2_LR;
        if (p3_LR < minLR) minLR = p3_LR;

        uint16_t aggLR = sat12_sw(curCost[d] + minLR - minPrevLR);
        aggLR_arr[d] = aggLR;

        uint16_t p0_TB = prevCostT_col[d];
        uint16_t p1_TB = (d > 0) ? sat12_sw(prevCostT_col[d - 1] + P1_SW) : INF_COST_SW;
        uint16_t p2_TB = (d < DISP - 1) ? sat12_sw(prevCostT_col[d + 1] + P1_SW) : INF_COST_SW;
        uint16_t p3_TB = sat12_sw(minPrevTB + P2_SW);

        uint16_t minTB = p0_TB;
        if (p1_TB < minTB) minTB = p1_TB;
        if (p2_TB < minTB) minTB = p2_TB;
        if (p3_TB < minTB) minTB = p3_TB;

        uint16_t aggTB = sat12_sw(curCost[d] + minTB - minPrevTB);
        aggTB_arr[d] = aggTB;

        if(aggLR < runMinLR) runMinLR = aggLR;
        if(aggTB < runMinTB) runMinTB = aggTB;

        uint16_t sum2 = sat12_sw(aggLR + aggTB);
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

void commit_prev_costs_sw(
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

void sgm_sw(
		const cv::Mat& left,
		const cv::Mat& right,
		cv::Mat& disp)
{
    disp = cv::Mat::zeros(IMG_H, IMG_W, CV_8UC1);
    uint8_t bufL[WIN][IMG_W];
    uint8_t bufR[WIN][IMG_W];

    for (int wy = 0; wy < WIN; ++wy)
    {
    	for (int c = 0; c < IMG_W; ++c)
    	{
    		bufL[wy][c] = 0;
            bufR[wy][c] = 0;
    	}
    }

    /* Cost arrays */
    uint16_t curCost[DISP];
    uint16_t prevCostL[DISP];
    uint16_t prevCostT[IMG_W][DISP];
    uint16_t aggCost[DISP];
    uint16_t aggLR_arr[DISP];
    uint16_t aggTB_arr[DISP];
    uint8_t leftWin[WIN][WIN];
    uint8_t rightStripe[WIN][RIGHT_STRIPE_W];
    uint16_t minPrevT[IMG_W];

    /* center offset */
    const int cx = WIN >> 1;


    for (int r = 0; r < IMG_H; r++)
    {
    	int right_wr = RIGHT_STRIPE_W - 1;
    	uint16_t minPrevLR = 0;

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
    		uint8_t pL = left.at<uint8_t>(r, c);
    		uint8_t pR = right.at<uint8_t>(r, c);

    		update_line_buffers_sw(bufL, bufR, c, pL, pR);

    		update_sliding_windows_sw(bufL, bufR, c, leftWin, rightStripe,
    				right_wr);

    		const bool interior = (r >= WIN - 1) && (c >= (DISP - 1) + 2* cx)
    				&& (c < IMG_W);

    	    if (interior)
    	    {
    	    	compute_sad_cost_vector_sw(leftWin, rightStripe, right_wr, curCost);
    	    }

    	    int out_c = c - cx;
    	    		if(out_c >= 0)
    	    		{
    	    			uint8_t outDisp = 0;
    	    			if(interior)
    	    			{
        	    			uint16_t newMinLR = INF_COST_SW;
        	    			uint16_t newMinTB = INF_COST_SW;

        	    			uint8_t bestDisp = aggregate_paths_and_select_sw(curCost,
        	    					prevCostL, prevCostT[out_c], minPrevLR,
									minPrevT[out_c], aggLR_arr, aggTB_arr,
									aggCost, newMinLR, newMinTB);

        	    	        commit_prev_costs_sw(prevCostL, prevCostT[out_c], aggLR_arr,
        	    	        		aggTB_arr);

        	    	        minPrevLR = newMinLR;
        	    	        minPrevT[out_c] = newMinTB;

        	    	        outDisp = bestDisp;
    	    			}
    	    	        disp.at<uint8_t>(r, out_c) = outDisp;
    	    		}
    	    	}
    	        for (int t = 0; t < cx; ++t)
    	        {
    	        	disp.at<uint8_t>(r, IMG_W - cx + t) = 0;
    	        }
    	}
}
