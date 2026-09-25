#include "sgm_params.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

static const cost_t INF_COST = cost_t(4095);

/* --------------------------------------------------------- */
/* Helper Function                                           */
/* --------------------------------------------------------- */

static inline void update_line_buffers(
		pix_t bufL[WIN][IMG_W],
		pix_t bufR[WIN][IMG_W],
		int c,
		pix_t pL,
		pix_t pR)
{
#pragma HLS INLINE

    for (int i = 0; i < WIN - 1; ++i)
    {
        bufL[i][c] = bufL[i + 1][c];
        bufR[i][c] = bufR[i + 1][c];
    }
    bufL[WIN - 1][c] = pL;
    bufR[WIN - 1][c] = pR;
}

static void update_sliding_windows(
		pix_t bufL[WIN][IMG_W],
		pix_t bufR[WIN][IMG_W],
		int c,
		pix_t leftWin[WIN][WIN],
		pix_t rightStripe[WIN][RIGHT_STRIPE_W],
		int& right_wr)
{
#pragma HLS INLINE

	ShiftLeftWin:
	    for (int wy = 0; wy < WIN; ++wy)
	    {
		#pragma HLS UNROLL
	        for (int wx = 0; wx < WIN - 1; ++wx)
	        {
		#pragma HLS UNROLL
	            leftWin[wy][wx] = leftWin[wy][wx + 1];
	        }
	    }

	InsertLeftCol:
		for (int wy = 0; wy < WIN; ++wy)
		{
	    #pragma HLS UNROLL
			leftWin[wy][WIN - 1] = bufL[wy][c];
	    }

		right_wr++;
		if(right_wr == RIGHT_STRIPE_W)
			right_wr = 0;

	ShiftRightStripe:
		for (int wy = 0; wy < WIN; ++wy)
		{
		#pragma HLS UNROLL
			rightStripe[wy][right_wr] = bufR[wy][c];
		}
}

census_t compute_census_descriptor(pix_t leftWin[WIN][WIN])
{
	census_t descriptor = 0;
	pix_t centerL = leftWin[CENSUS_CY][CENSUS_CX];

    CENSUS_WinY:
        for (int wy = 0; wy < WIN; ++wy)
        {
		#pragma HLS UNROLL

        CENSUS_WinX:
            for (int wx = 0; wx < WIN; ++wx)
            {
			#pragma HLS UNROLL
            	if (wy == CENSUS_CY && wx == CENSUS_CX)
            	    continue;

            	descriptor <<= 1;
				descriptor[0] = leftWin[wy][wx] < centerL;
            }
        }
        return descriptor;
}

census_t compute_right_census_descriptor(pix_t rightStripe
		[WIN][RIGHT_STRIPE_W], int right_wr)
{
	census_t descriptor = 0;

	int center_idx = right_wr - CENSUS_CX;
	if(center_idx < 0)
		center_idx += RIGHT_STRIPE_W;

	pix_t centerR = rightStripe[CENSUS_CY][center_idx];

    CENSUS_WinY:
        for (int wy = 0; wy < WIN; ++wy)
        {
		#pragma HLS UNROLL

        CENSUS_WinX:
            for (int wx = 0; wx < WIN; ++wx)
            {
			#pragma HLS UNROLL
            	if (wy == CENSUS_CY && wx == CENSUS_CX)
            	    continue;

					int physIndex = right_wr - (WIN - 1 - wx);

					if (physIndex < 0)
							physIndex += RIGHT_STRIPE_W;

					descriptor <<= 1;

					descriptor[0] = rightStripe[wy][physIndex] < centerR;
            }
        }
        return descriptor;
}

static void compute_census_cost_vector(
		census_t leftDesc,
		census_t rightCensusHistory[DISP],
	    cost_t curCost[DISP])
{
#pragma HLS INLINE off

	CENSUS_Disparity:
	for (int d = 0; d < DISP; ++d)
	{
	#pragma HLS PIPELINE II=1
		census_t diff = leftDesc ^ rightCensusHistory[d];
		cost_t sum  = 0;

		for(int b = 0; b < 8; ++b)
		{
		#pragma HLS UNROLL
			sum += diff[b];
		}
        curCost[d] = sum;
	}
}

static disp_t aggregate_paths_and_select(
    const cost_t curCost[DISP],
    const cost_t prevCostL[DISP],
    const cost_t prevCostT_col[DISP],
    cost_t minPrevLR,
    cost_t minPrevTB,
    cost_t aggLR_arr[DISP],
    cost_t aggTB_arr[DISP],
    cost_t aggCost[DISP],
	cost_t& newMinLR,
	cost_t& newMinTB)
{
#pragma HLS INLINE off

    cost_t bestCost = INF_COST;
    disp_t bestDisp = 0;

    cost_t runMinLR = INF_COST;
    cost_t runMinTB = INF_COST;

AggregationLoop:
    for (int d = 0; d < DISP; d++)
    {
	#pragma HLS PIPELINE II = 1
        cost_t p0_LR = prevCostL[d];
        cost_t p1_LR = (d > 0) ? sat12(prevCostL[d - 1] + P1) : INF_COST;
        cost_t p2_LR = (d < DISP - 1) ? sat12(prevCostL[d + 1] + P1) : INF_COST;
        cost_t p3_LR = sat12(minPrevLR + P2);

        cost_t minLR = p0_LR;
        if (p1_LR < minLR) minLR = p1_LR;
        if (p2_LR < minLR) minLR = p2_LR;
        if (p3_LR < minLR) minLR = p3_LR;

        cost_t aggLR = sat12(curCost[d] + minLR - minPrevLR);
        aggLR_arr[d] = aggLR;

        cost_t p0_TB = prevCostT_col[d];
        cost_t p1_TB = (d > 0) ? sat12(prevCostT_col[d - 1] + P1) : INF_COST;
        cost_t p2_TB = (d < DISP - 1) ? sat12(prevCostT_col[d + 1] + P1) : INF_COST;
        cost_t p3_TB = sat12(minPrevTB + P2);

        cost_t minTB = p0_TB;
        if (p1_TB < minTB) minTB = p1_TB;
        if (p2_TB < minTB) minTB = p2_TB;
        if (p3_TB < minTB) minTB = p3_TB;

        cost_t aggTB = sat12(curCost[d] + minTB - minPrevTB);
        aggTB_arr[d] = aggTB;

        if(aggLR < runMinLR) runMinLR = aggLR;
        if(aggTB < runMinTB) runMinTB = aggTB;

        cost_t sum2 = sat12(aggLR + aggTB);
        aggCost[d] = sum2;

        if (sum2 < bestCost)
        {
            bestCost = sum2;
            bestDisp = disp_t(d);
        }
    }
    newMinLR = runMinLR;
    newMinTB = runMinTB;

    return bestDisp;
}

static void commit_prev_costs(
    cost_t prevCostL[DISP],
    cost_t prevCostT_col[DISP],
    const cost_t aggLR_arr[DISP],
    const cost_t aggTB_arr[DISP])
{
#pragma HLS INLINE off

CopyPrevLR:
    for (int d = 0; d < DISP; ++d)
    {
	#pragma HLS UNROLL
        prevCostL[d]    = aggLR_arr[d];
        prevCostT_col[d] = aggTB_arr[d];
    }
}
struct CostPacket
{
	bool valid;
	cost_t curCost[DISP];
};

static CostPacket col_frontend(
	    bram_word_t left[FRAME_WORDS],
		bram_word_t right[FRAME_WORDS],
		pix_t bufL[WIN][IMG_W],
		pix_t bufR[WIN][IMG_W],
	    int r,
	    int c,
	    int cx,
		pix_t leftWin[WIN][WIN],
		pix_t rightStripe[WIN][RIGHT_STRIPE_W],
		census_t rightCensusHistory[DISP],
		int& right_wr)
{
#pragma HLS INLINE off
	CostPacket pkt;
	pkt.valid = false;

    int pixel_idx = r * IMG_W + c;
    int word_indx = pixel_idx >> 2;
    int byte_idx = pixel_idx & 3;

    bram_word_t left_word = left[word_indx];
    pix_t pL = pix_t(left_word >> (byte_idx * 8));

    bram_word_t right_word = right[word_indx];
    pix_t pR = pix_t(right_word >> (byte_idx * 8));

	update_line_buffers(bufL, bufR, c, pL, pR);
	update_sliding_windows(bufL, bufR, c, leftWin, rightStripe,
			right_wr);

	census_t leftDesc = compute_census_descriptor(leftWin);
	census_t newRightDesc = compute_right_census_descriptor
	(rightStripe, right_wr);

	for (int d = DISP - 1; d > 0; --d)
	{
	#pragma HLS UNROLL
	    rightCensusHistory[d] =
	        rightCensusHistory[d - 1];
	}

	rightCensusHistory[0] = newRightDesc;

	const bool interior =
	    (r >= WIN - 1) &&
	    (c >= (DISP - 1) + 2* cx) &&
	    (c < IMG_W);

    if (interior)
    {
    	compute_census_cost_vector(leftDesc, rightCensusHistory,
    			pkt.curCost);
    	pkt.valid = true;
    }
    else
    {
    	pkt.valid = false;
    }
    return pkt;
}

static disp_t col_backend(
		const CostPacket& pkt,
		cost_t prevCostL[DISP],
		cost_t prevCostT_col[DISP],
		cost_t aggLR_arr[DISP],
		cost_t aggTB_arr[DISP],
		cost_t aggCost[DISP],
		cost_t& minPrevLR,
		cost_t& minPrevTB)
{
#pragma HLS INLINE off
	disp_t outDisp = 0;

	if (pkt.valid)
	{
        cost_t newMinLR = INF_COST;
        cost_t newMinTB = INF_COST;

        disp_t bestDisp = aggregate_paths_and_select(
            pkt.curCost,
            prevCostL,
			prevCostT_col,
            minPrevLR,
            minPrevTB,
            aggLR_arr,
            aggTB_arr,
            aggCost,
			newMinLR,
			newMinTB);

        commit_prev_costs(
            prevCostL,
			prevCostT_col,
            aggLR_arr,
            aggTB_arr);

        minPrevLR = newMinLR;
        minPrevTB = newMinTB;
        outDisp = bestDisp;
	}
    return outDisp;
}

/* --------------------------------------------------------- */
/* Top kernel                                                */
/* --------------------------------------------------------- */

void sgm_kernel(bram_word_t left[FRAME_WORDS],
				bram_word_t right[FRAME_WORDS],
				bram_word_t disp[FRAME_WORDS])
{
#pragma HLS INTERFACE mode=bram		port=left
#pragma HLS INTERFACE mode=bram		port=right
#pragma HLS INTERFACE mode=bram		port=disp

#pragma HLS INTERFACE mode=s_axilite	port=return	bundle=control


    /* Line buffers for the left & right images */
    pix_t bufL[WIN][IMG_W];
    pix_t bufR[WIN][IMG_W];
#pragma HLS ARRAY_PARTITION variable=bufL complete dim=1
#pragma HLS ARRAY_PARTITION variable=bufR complete dim=1

    InitBuf:
    for (int wy = 0; wy < WIN; ++wy)
    {
        for (int c = 0; c < IMG_W; ++c)
        {
            bufL[wy][c] = 0;
            bufR[wy][c] = 0;
        }
    }

    /* Cost arrays */
    static cost_t prevCostL[DISP];
#pragma HLS ARRAY_PARTITION variable=prevCostL complete dim=1

    static cost_t prevCostT[IMG_W][DISP];
#pragma HLS bind_storage variable=prevCostT type=RAM_2P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=prevCostT complete dim=2

    static cost_t aggCost[DISP];
#pragma HLS ARRAY_PARTITION variable=aggCost complete dim=1

    static cost_t aggLR_arr[DISP];
    static cost_t aggTB_arr[DISP];
#pragma HLS ARRAY_PARTITION variable=aggLR_arr complete dim=1
#pragma HLS ARRAY_PARTITION variable=aggTB_arr complete dim=1

    pix_t leftWin[WIN][WIN];
    pix_t rightStripe[WIN][RIGHT_STRIPE_W];

#pragma HLS ARRAY_PARTITION variable=leftWin complete dim=0
#pragma HLS ARRAY_PARTITION variable=rightStripe complete dim=1

    census_t rightCensusHistory[DISP];
#pragma HLS ARRAY_PARTITION variable=rightCensusHistory complete

    static cost_t minPrevT[IMG_W];

    /* center offset */
    const int cx = WIN >> 1;

Row:
    for (int r = 0; r < IMG_H; r++)
    {
    	int right_wr = RIGHT_STRIPE_W - 1;

    	cost_t minPrevLR = 0;

    	bram_word_t disp_word = 0;

    	for (int d = 0; d < DISP; ++d)
    	{
		#pragma HLS UNROLL factor=2
    	    rightCensusHistory[d] = 0;
    	}

        /* Reset aggregation for new row */
    ResetCosts:
        for (int d = 0; d < DISP; d++)
        {
		#pragma HLS UNROLL factor=2
            prevCostL[d] = cost_t(0);
        }

        if(r == 0)
        {
        InitTBRow:
			for (int c = 0; c < IMG_W; ++c)
			{
			#pragma HLS LOOP_TRIPCOUNT min=IMG_W max=IMG_W
		        minPrevT[c] = cost_t(0);
				InitTBRowD:
				for (int d = 0; d < DISP; ++d)
				{
				#pragma HLS UNROLL factor=2
					prevCostT[c][d] = cost_t(0);
				}
			}
        }

    	InitLeftWin:
    	for (int wy = 0; wy < WIN; ++wy)
    	{
    	    for (int wx = 0; wx < WIN; ++wx)
    	    {
    	        leftWin[wy][wx] = 0;
    	    }
    	}

    	InitRightStripe:
    	for (int wy = 0; wy < WIN; ++wy)
    	{
    	    for (int k = 0; k < RIGHT_STRIPE_W; ++k)
    	    {
    	        rightStripe[wy][k] = 0;
    	    }
    	}

    	for (int c = 0; c < IMG_W; ++c)
    	{
    	//#pragma HLS PIPELINE II=16
    	#pragma HLS DEPENDENCE variable=bufL inter false
    	#pragma HLS DEPENDENCE variable=bufR inter false

    		CostPacket pkt = col_frontend(
    				left,
					right,
					bufL,
					bufR,
					r,
					c,
					cx,
    				leftWin,
					rightStripe,
					rightCensusHistory,
					right_wr);

    		int out_c = c - cx;
    		if(out_c >= 0)
    		{
    			disp_t outDisp = col_backend(
    					pkt,
    					prevCostL,
						prevCostT[out_c],
						aggLR_arr,
						aggTB_arr,
						aggCost,
						minPrevLR,
						minPrevT[out_c]);

    				int disp_pixel_indx = r * IMG_W + out_c;
    				int disp_word_indx = disp_pixel_indx >> 2;
    				int disp_byte_indx = disp_pixel_indx & 3;

    				disp_word.range(disp_byte_indx * 8 + 7,
    						disp_byte_indx * 8) = outDisp;

    				if(disp_byte_indx == 3)
    				{
    					disp[disp_word_indx] = disp_word;
    					disp_word = 0;
    				}
    		}
    	}
    	int last_pixel_idx = r * IMG_W + (IMG_W - 1);
    	int last_word_idx  = last_pixel_idx >> 2;

    	disp_word.range(31, 24) = 0;
    	disp[last_word_idx] = disp_word;
    }
}
