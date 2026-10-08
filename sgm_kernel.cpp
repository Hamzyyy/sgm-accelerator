#include "sgm_params.hpp"
#include <hls_stream.h>

#ifndef __SYNTHESIS__
#include <iostream>
#endif

static const cost_t INF_COST = cost_t(4095);
static const wide_t INF_WIDE = wide_t(16383);

struct CostPacket
{
	bool valid;
	cost_t curCost[PAR];
};


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
		int g,
		int out_c,
	    cost_t curCost[PAR])
{
#pragma HLS INLINE
		for(int lane = 0; lane < PAR; ++lane)
		{
		#pragma HLS UNROLL
			int d = g * PAR + lane;

			if(out_c >= d)
			{
				census_t diff = leftDesc ^ rightCensusHistory[d];
				cost_t sum  = 0;

				for(int b = 0; b < 8; ++b)
				{
				#pragma HLS UNROLL
					sum += diff[b];
				}
		        curCost[lane] = sum;
			}
			else
			{
				curCost[lane] = INF_COST;
			}
		}
}
//////////////////////////////////////////////////////
static void aggregate_group(const CostPacket& pkt,
							int g,
							const cost_t prevCostL[DISP],
							const cost_t prevCostT_col[DISP],
							const cost_t carryTL[DISP],
							const cost_t prevCostTR_col_pred[DISP],
							cost_t minPrevLR,
							cost_t minPrevTB,
							cost_t carryMinTL,
							cost_t minPrevTR_pred,
							cost_t aggLR_arr[DISP],
							cost_t aggTB_arr[DISP],
							cost_t aggTL_arr[DISP],
							cost_t aggTR_arr[DISP],
							cost_t& groupMinLR,
							cost_t& groupMinTB,
							cost_t& groupMinTL,
							cost_t& groupMinTR,
							wide_t& groupBestCost,
							disp_t& groupBestDisp)
{
#pragma HLS INLINE

    cost_t laneAggLR[PAR];
    cost_t laneAggTB[PAR];
    cost_t laneAggTL[PAR];
    cost_t laneAggTR[PAR];
    wide_t laneSum[PAR];

#pragma HLS ARRAY_PARTITION variable=laneAggLR type=complete
#pragma HLS ARRAY_PARTITION variable=laneAggTB type=complete
#pragma HLS ARRAY_PARTITION variable=laneAggTL type=complete
#pragma HLS ARRAY_PARTITION variable=laneAggTR type=complete
#pragma HLS ARRAY_PARTITION variable=laneSum type=complete

	LaneLoop:
	    	for(int lane = 0; lane < PAR; lane++)
	    	{
			#pragma HLS UNROLL
	    		int d = g * PAR + lane;

				cost_t p0_LR = prevCostL[d];
				cost_t p1_LR = (d > 0) ? sat12(prevCostL[d - 1] + P1) : INF_COST;
				cost_t p2_LR = (d < DISP - 1) ? sat12(prevCostL[d + 1] + P1)
						: INF_COST;
				cost_t p3_LR = sat12(minPrevLR + P2);

				cost_t minLR = p0_LR;
				if (p1_LR < minLR) minLR = p1_LR;
				if (p2_LR < minLR) minLR = p2_LR;
				if (p3_LR < minLR) minLR = p3_LR;

				cost_t aggLR = sat12(pkt.curCost[lane] + minLR - minPrevLR);
				aggLR_arr[d] = aggLR;
				laneAggLR[lane] = aggLR;




				cost_t p0_TB = prevCostT_col[d];
				cost_t p1_TB = (d > 0) ? sat12(prevCostT_col[d - 1] + P1)
						: INF_COST;
				cost_t p2_TB = (d < DISP - 1) ? sat12(prevCostT_col[d + 1] + P1)
						: INF_COST;
				cost_t p3_TB = sat12(minPrevTB + P2);

				cost_t minTB = p0_TB;
				if (p1_TB < minTB) minTB = p1_TB;
				if (p2_TB < minTB) minTB = p2_TB;
				if (p3_TB < minTB) minTB = p3_TB;

				cost_t aggTB = sat12(pkt.curCost[lane] + minTB - minPrevTB);
				aggTB_arr[d] = aggTB;
				laneAggTB[lane] = aggTB;






				cost_t p0_TL = carryTL[d];
				cost_t p1_TL = (d > 0) ? sat12(carryTL[d - 1] + P1)
						: INF_COST;
				cost_t p2_TL = (d < DISP - 1) ? sat12(carryTL[d + 1] + P1)
						: INF_COST;
				cost_t p3_TL = sat12(carryMinTL + P2);

				cost_t minTL = p0_TL;
				if (p1_TL < minTL) minTL = p1_TL;
				if (p2_TL < minTL) minTL = p2_TL;
				if (p3_TL < minTL) minTL = p3_TL;

				cost_t aggTL = sat12(pkt.curCost[lane] + minTL - carryMinTL);
				aggTL_arr[d] = aggTL;
				laneAggTL[lane] = aggTL;





				cost_t p0_TR = prevCostTR_col_pred[d];
				cost_t p1_TR = (d > 0) ? sat12(prevCostTR_col_pred[d - 1] + P1)
						: INF_COST;
				cost_t p2_TR = (d < DISP - 1) ? sat12(prevCostTR_col_pred[d + 1] + P1)
						: INF_COST;
				cost_t p3_TR = sat12(minPrevTR_pred + P2);

				cost_t minTR = p0_TR;
				if (p1_TR < minTR) minTR = p1_TR;
				if (p2_TR < minTR) minTR = p2_TR;
				if (p3_TR < minTR) minTR = p3_TR;

				cost_t aggTR = sat12(pkt.curCost[lane] + minTR - minPrevTR_pred);
				aggTR_arr[d] = aggTR;
				laneAggTR[lane] = aggTR;


				pair_t sum1 = pair_t(aggLR) + pair_t(aggTB);
				pair_t sum2 = pair_t(aggTL) + pair_t(aggTR);

				wide_t sum = wide_t(sum1) + wide_t(sum2);

				laneSum[lane]   = sum;
	    	}


	    	cost_t minLR01 = (laneAggLR[1] < laneAggLR[0]) ? laneAggLR[1]
							: laneAggLR[0];
	    	cost_t minLR23 = (laneAggLR[3] < laneAggLR[2]) ? laneAggLR[3]
							: laneAggLR[2];
	    	groupMinLR = (minLR23 < minLR01) ? minLR23 : minLR01;




	    	cost_t minTB01 = (laneAggTB[1] < laneAggTB[0]) ? laneAggTB[1]
							: laneAggTB[0];
	    	cost_t minTB23 = (laneAggTB[3] < laneAggTB[2]) ? laneAggTB[3]
							: laneAggTB[2];
	    	groupMinTB = (minTB23 < minTB01) ? minTB23 : minTB01;






	    	cost_t minTL01 = (laneAggTL[1] < laneAggTL[0]) ? laneAggTL[1]
							: laneAggTL[0];
	    	cost_t minTL23 = (laneAggTL[3] < laneAggTL[2]) ? laneAggTL[3]
							: laneAggTL[2];
	    	groupMinTL = (minTL23 < minTL01) ? minTL23 : minTL01;




	    	cost_t minTR01 = (laneAggTR[1] < laneAggTR[0]) ? laneAggTR[1]
							: laneAggTR[0];
	    	cost_t minTR23 = (laneAggTR[3] < laneAggTR[2]) ? laneAggTR[3]
							: laneAggTR[2];
	    	groupMinTR = (minTR23 < minTR01) ? minTR23 : minTR01;






	    	wide_t bestCost01;
	    	disp_t bestDisp01;

	    	if (laneSum[1] < laneSum[0])
	    	{
	    	    bestCost01 = laneSum[1];
	    	    bestDisp01 = disp_t(g * PAR + 1);
	    	}
	    	else
	    	{
	    	    bestCost01 = laneSum[0];
	    	    bestDisp01 = disp_t(g * PAR);
	    	}

	    	wide_t bestCost23;
	    	disp_t bestDisp23;

	    	if (laneSum[3] < laneSum[2])
	    	{
	    	    bestCost23 = laneSum[3];
	    	    bestDisp23 = disp_t(g * PAR + 3);
	    	}
	    	else
	    	{
	    	    bestCost23 = laneSum[2];
	    	    bestDisp23 = disp_t(g * PAR + 2);
	    	}

	    	if (bestCost23 < bestCost01)
	    	{
	    	    groupBestCost = bestCost23;
	    	    groupBestDisp = bestDisp23;
	    	}
	    	else
	    	{
	    	    groupBestCost = bestCost01;
	    	    groupBestDisp = bestDisp01;
	    	}
}
//////////////////////////////////////////////////////

static void commit_prev_costs(cost_t prevCostL[DISP],
							cost_t prevCostT_col[DISP],
							cost_t prevCostTL_col[DISP],
							cost_t prevCostTR_col[DISP],
							const cost_t aggLR_arr[DISP],
							const cost_t aggTB_arr[DISP],
							const cost_t aggTL_arr[DISP],
							const cost_t aggTR_arr[DISP],
							cost_t carryTL[DISP])
{
#pragma HLS INLINE off

CopyPrevLR:
    for (int d = 0; d < DISP; ++d)
    {
	#pragma HLS UNROLL
        prevCostL[d]    = aggLR_arr[d];
        prevCostT_col[d] = aggTB_arr[d];
        prevCostTR_col[d] = aggTR_arr[d];
        carryTL[d] = prevCostTL_col[d];
        prevCostTL_col[d] = aggTL_arr[d];
    }
}


static bool prepare_census_column(
	    bram_word_t left[FRAME_WORDS],
		bram_word_t right[FRAME_WORDS],
		pix_t bufL[WIN][IMG_W],
		pix_t bufR[WIN][IMG_W],
	    int r,
	    int c,
	    int cx,
		pix_t leftWin[WIN][WIN],
		pix_t rightStripe[WIN][RIGHT_STRIPE_W],
		census_t &leftDesc,
		census_t rightCensusHistory[DISP],
		int& right_wr)
{
#pragma HLS INLINE off

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

	leftDesc = compute_census_descriptor(leftWin);
	census_t newRightDesc = compute_right_census_descriptor(rightStripe,
			right_wr);

	for (int d = DISP - 1; d > 0; --d)
	{
	#pragma HLS UNROLL
	    rightCensusHistory[d] =
	        rightCensusHistory[d - 1];
	}

	rightCensusHistory[0] = newRightDesc;
	const bool interior = (r >= WIN - 1) && (c < IMG_W);

	return interior;
}



static void col_frontend(
		hls::stream<CostPacket>& coststream,
		bool interior,
		int out_c,
	    census_t leftDesc,
		census_t rightCensusHistory[DISP])
{
#pragma HLS INLINE off
	FrontendGroups:
	    for (int g = 0; g < GROUPS; ++g)
	    {
		#pragma HLS PIPELINE II=1
	    	CostPacket pkt;
	        pkt.valid = interior;

	        if (interior)
	        {
	            compute_census_cost_vector(
	                leftDesc,
	                rightCensusHistory,
	                g,
					out_c,
	                pkt.curCost);
	        }
	        coststream.write(pkt);
	    }
}



static disp_t col_backend(hls::stream<CostPacket>& coststream,
						cost_t prevCostL[DISP],
						cost_t prevCostT_col[DISP],
						cost_t prevCostTL_col[DISP],
						cost_t prevCostTR_col_pred[DISP],
						cost_t prevCostTR_col_cur[DISP],
						cost_t aggLR_arr[DISP],
						cost_t aggTB_arr[DISP],
						cost_t aggTL_arr[DISP],
						cost_t aggTR_arr[DISP],
						cost_t carryTL[DISP],
						cost_t& minPrevLR,
						cost_t& minPrevTB,
						cost_t& minPrevTL,
						cost_t& minPrevTR_pred,
						cost_t& minPrevTR_cur,
						cost_t& carryMinTL)
{
#pragma HLS INLINE off

    cost_t runMinLR = INF_COST;
    cost_t runMinTB = INF_COST;
    cost_t runMinTL = INF_COST;
    cost_t runMinTR = INF_COST;

    wide_t bestCost = INF_WIDE;
    disp_t bestDisp = 0;

    bool validPixel = false;

    BackendGroups:
        for (int g = 0; g < GROUPS; ++g)
        {
		#pragma HLS PIPELINE II=1
        	CostPacket pkt = coststream.read();

            cost_t groupMinLR;
            cost_t groupMinTB;
            cost_t groupMinTL;
            cost_t groupMinTR;
            wide_t groupBestCost;
            disp_t groupBestDisp;

            if(pkt.valid)
            {
            	validPixel = true;
                aggregate_group(pkt,
								g,
								prevCostL,
								prevCostT_col,
								carryTL,
								prevCostTR_col_pred,
								minPrevLR,
								minPrevTB,
								carryMinTL,
								minPrevTR_pred,
								aggLR_arr,
								aggTB_arr,
								aggTL_arr,
								aggTR_arr,
								groupMinLR,
								groupMinTB,
								groupMinTL,
								groupMinTR,
								groupBestCost,
								groupBestDisp);

                if (groupMinLR < runMinLR)
                    runMinLR = groupMinLR;

                if (groupMinTB < runMinTB)
                    runMinTB = groupMinTB;

                if (groupMinTL < runMinTL)
                    runMinTL = groupMinTL;

                if (groupMinTR < runMinTR)
                    runMinTR = groupMinTR;

                if (groupBestCost < bestCost)
                {
                    bestCost = groupBestCost;
                    bestDisp = groupBestDisp;
                }
            }
        }
        if(validPixel)
        {
            commit_prev_costs(prevCostL, prevCostT_col, prevCostTL_col,
            		prevCostTR_col_cur, aggLR_arr, aggTB_arr,
					aggTL_arr, aggTR_arr, carryTL);

            minPrevLR = runMinLR;
            minPrevTB = runMinTB;
            minPrevTR_cur = runMinTR;
            carryMinTL = minPrevTL;
            minPrevTL = runMinTL;
        }
        return bestDisp;
}



static disp_t process_cost_groups(
    bool interior,
    census_t leftDesc,
    census_t rightCensusHistory[DISP],
    cost_t prevCostL[DISP],
    cost_t prevCostT_col[DISP],
	cost_t prevCostTL_col[DISP],
	cost_t prevCostTR_col_pred[DISP],
	cost_t prevCostTR_col_cur[DISP],
    cost_t aggLR_arr[DISP],
    cost_t aggTB_arr[DISP],
    cost_t aggTL_arr[DISP],
    cost_t aggTR_arr[DISP],
    cost_t carryTL[DISP],
    cost_t& minPrevLR,
    cost_t& minPrevTB,
    cost_t& minPrevTL,
	cost_t& minPrevTR_pred,
	cost_t& minPrevTR_cur,
	cost_t& carryMinTL,
	int out_c)
{
#pragma HLS DATAFLOW

    hls::stream<CostPacket> coststream;
#pragma HLS STREAM variable=coststream depth=2

    col_frontend(
        coststream,
        interior,
		out_c,
        leftDesc,
        rightCensusHistory);

    return col_backend(coststream,
					prevCostL,
					prevCostT_col,
					prevCostTL_col,
					prevCostTR_col_pred,
					prevCostTR_col_cur,
					aggLR_arr,
					aggTB_arr,
					aggTL_arr,
					aggTR_arr,
					carryTL,
					minPrevLR,
					minPrevTB,
					minPrevTL,
					minPrevTR_pred,
					minPrevTR_cur,
					carryMinTL);
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

    static cost_t prevCostTL[IMG_W][DISP];
#pragma HLS bind_storage variable=prevCostTL type=RAM_2P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=prevCostTL complete dim=2

    static cost_t carryTL[DISP];
#pragma HLS ARRAY_PARTITION variable=carryTL complete dim=1

    static cost_t prevCostTR[IMG_W][DISP];
#pragma HLS bind_storage variable=prevCostTR type=RAM_2P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=prevCostTR complete dim=2

    static cost_t aggLR_arr[DISP];
    static cost_t aggTB_arr[DISP];
    static cost_t aggTL_arr[DISP];
    static cost_t aggTR_arr[DISP];
#pragma HLS ARRAY_PARTITION variable=aggLR_arr complete dim=1
#pragma HLS ARRAY_PARTITION variable=aggTB_arr complete dim=1
#pragma HLS ARRAY_PARTITION variable=aggTL_arr complete dim=1
#pragma HLS ARRAY_PARTITION variable=aggTR_arr complete dim=1

    pix_t leftWin[WIN][WIN];
    pix_t rightStripe[WIN][RIGHT_STRIPE_W];

#pragma HLS ARRAY_PARTITION variable=leftWin complete dim=0
#pragma HLS ARRAY_PARTITION variable=rightStripe complete dim=1

    census_t rightCensusHistory[DISP];
#pragma HLS ARRAY_PARTITION variable=rightCensusHistory complete

    static cost_t minPrevT[IMG_W];
    static cost_t minPrevTL[IMG_W];
    static cost_t minPrevTR[IMG_W];

    static cost_t carryMinTL;

    /* center offset */
    const int cx = WIN >> 1;
    const int cy = WIN >> 1;

Row:
    for (int r = 0; r < IMG_H; r++)
    {
    	int right_wr = RIGHT_STRIPE_W - 1;

    	cost_t minPrevLR = 0;

    	bram_word_t disp_word = 0;

    	carryMinTL = 0;

    	int out_r = r - cy;

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
            carryTL[d] = cost_t(0);
        }

        if(r == 0)
        {
        InitTRow:
			for (int c = 0; c < IMG_W; ++c)
			{
			#pragma HLS LOOP_TRIPCOUNT min=IMG_W max=IMG_W
		        minPrevT[c] = cost_t(0);
		        minPrevTL[c] = cost_t(0);
		        minPrevTR[c] = cost_t(0);
				InitTBRowD:
				for (int d = 0; d < DISP; ++d)
				{
				#pragma HLS UNROLL factor=2
					prevCostT[c][d] = cost_t(0);
					prevCostTL[c][d] = cost_t(0);
					prevCostTR[c][d] = cost_t(0);
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

    	census_t leftDesc;

    	for (int c = 0; c < IMG_W; ++c)
    	{
    	//#pragma HLS PIPELINE II=16
    	#pragma HLS DEPENDENCE variable=bufL inter false
    	#pragma HLS DEPENDENCE variable=bufR inter false

    		bool interior = prepare_census_column(
    		        left,
    		        right,
    		        bufL,
    		        bufR,
    		        r,
    		        c,
    		        cx,
    		        leftWin,
    		        rightStripe,
    		        leftDesc,
    		        rightCensusHistory,
    		        right_wr);

    		int out_c = c - cx;
    		if(out_r >= 0 && out_c >= 0)
    		{
    			disp_t outDisp = process_cost_groups(interior,
													leftDesc,
													rightCensusHistory,
													prevCostL,
													prevCostT[out_c],
													prevCostTL[out_c],
													prevCostTR[out_c+1],
													prevCostTR[out_c],
													aggLR_arr,
													aggTB_arr,
													aggTL_arr,
													aggTR_arr,
													carryTL,
													minPrevLR,
													minPrevT[out_c],
													minPrevTL[out_c],
													minPrevTR[out_c+1],
													minPrevTR[out_c],
													carryMinTL,
													out_c);


    			int disp_pixel_indx = out_r * IMG_W + out_c;
    			int disp_word_indx = disp_pixel_indx >> 2;
    			int disp_byte_indx = disp_pixel_indx & 3;

    			disp_word.range(disp_byte_indx * 8 + 7, disp_byte_indx * 8) =
    					outDisp;

    				if(disp_byte_indx == 3)
    				{
    					disp[disp_word_indx] = disp_word;
    					disp_word = 0;
    				}
    		}
    	}
    	if(out_r >= 0)
    	{
            int last_pixel_idx = out_r * IMG_W + (IMG_W - 1);
            int last_word_idx  = last_pixel_idx >> 2;

           	disp_word.range(31, 24) = 0;
           	disp[last_word_idx] = disp_word;
    	}
    }
}
