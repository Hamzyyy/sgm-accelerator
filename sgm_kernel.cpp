#include "sgm_params.hpp"
#include <hls_stream.h>

#ifndef __SYNTHESIS__
#include <iostream>
#endif

static const cost_t INF_COST = cost_t(63);

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
//////////////////////////////////////////////////////

struct PrepPacket
{
	bool valid;
	int g;
	cost_t minPrevLR;
	cost_t minPrevTB;
	cost_t curCost[PAR];
	cost_t lrCenter[PAR];
	cost_t lrLeft[PAR];
	cost_t lrRight[PAR];
	cost_t tbCenter[PAR];
	cost_t tbLeft[PAR];
	cost_t tbRight[PAR];

};
static void prepare_group(const CostPacket& pkt,
	    				int g,
						const cost_t prevCostL[DISP],
						group_t tbPrevWord,
						group_t tbCurWord,
						group_t tbNextWord,
						cost_t minPrevLR,
						cost_t minPrevTB,
						PrepPacket& out)
{
	out.valid = pkt.valid;
	out.g = g;
	out.minPrevLR = minPrevLR;
	out.minPrevTB = minPrevTB;

    cost_t tbCur[PAR];
#pragma HLS ARRAY_PARTITION variable=tbCur type=complete

    for(int lane = 0; lane < PAR; ++lane)
    {
	#pragma HLS UNROLL
    	tbCur[lane] = tbCurWord.range((lane+1) * COST_BITS - 1,
    			lane * COST_BITS);
    }

    cost_t leftBoundary =
        tbPrevWord.range(PAR * COST_BITS - 1,
                         (PAR - 1) * COST_BITS);

    cost_t rightBoundary =
        tbNextWord.range(COST_BITS - 1, 0);

	LaneLoop:
	    	for(int lane = 0; lane < PAR; lane++)
	    	{
			#pragma HLS UNROLL
	    		int d = g * PAR + lane;

	    		out.curCost[lane] = pkt.curCost[lane];

				out.lrCenter[lane] = prevCostL[d];
				out.lrLeft[lane] = (d > 0) ? prevCostL[d - 1] : INF_COST;
				out.lrRight[lane] = (d < DISP - 1) ? prevCostL[d + 1] : INF_COST;


				out.tbCenter[lane] = tbCur[lane];
				if(lane == 0)
					out.tbLeft[lane] = (d > 0) ? leftBoundary : INF_COST;
				else
					out.tbLeft[lane] = (d > 0) ? tbCur[lane - 1] : INF_COST;

				if(lane == PAR-1)
					out.tbRight[lane] = (d < DISP - 1) ? rightBoundary : INF_COST;
				else
					out.tbRight[lane] = (d < DISP - 1) ? tbCur[lane+1] : INF_COST;
	    	}
}



struct AggPacket
{
    bool valid;
    int g;

    cost_t aggLR[PAR];
    cost_t aggTB[PAR];
};



static void aggregate_group(const PrepPacket& in,
							AggPacket& out)
{
	out.valid = in.valid;
	out.g = in.g;

	LaneLoop:
	    	for(int lane = 0; lane < PAR; lane++)
	    	{
			#pragma HLS UNROLL

				cost_t p0_LR = in.lrCenter[lane];
				cost_t p1_LR = sat12(wide_t(in.lrLeft[lane]) + wide_t(P1));
				cost_t p2_LR = sat12(wide_t(in.lrRight[lane]) + wide_t(P1));
				cost_t p3_LR = sat12(wide_t(in.minPrevLR) + wide_t(P2));

				cost_t minLR = p0_LR;
				if (p1_LR < minLR) minLR = p1_LR;
				if (p2_LR < minLR) minLR = p2_LR;
				if (p3_LR < minLR) minLR = p3_LR;

				if (in.curCost[lane] == INF_COST)
				    out.aggLR[lane] = INF_COST;
				else
					out.aggLR[lane] = sat12(wide_t(in.curCost[lane]) + wide_t(minLR)
							- wide_t(in.minPrevLR));

				cost_t p0_TB = in.tbCenter[lane];
				cost_t p1_TB = sat12(wide_t(in.tbLeft[lane]) + wide_t(P1));
				cost_t p2_TB = sat12(wide_t(in.tbRight[lane]) + wide_t(P1));


				cost_t p3_TB = sat12(wide_t(in.minPrevTB) + wide_t(P2));

				cost_t minTB = p0_TB;
				if (p1_TB < minTB) minTB = p1_TB;
				if (p2_TB < minTB) minTB = p2_TB;
				if (p3_TB < minTB) minTB = p3_TB;


				if (in.curCost[lane] == INF_COST)
					out.aggTB[lane] = INF_COST;
				else
					out.aggTB[lane] = sat12(wide_t(in.curCost[lane]) + wide_t(minTB)
							- wide_t(in.minPrevTB));
	    	}
}


struct ResultPacket
{
    bool valid;
    int g;

    cost_t groupMinLR;
    cost_t groupMinTB;

    cost_t groupBestCost;
    disp_t groupBestDisp;

    group_t newTBWord;

    cost_t aggLR[PAR];
};

static void reduce_group(const AggPacket& in,
						ResultPacket& out)
{
    out.valid = in.valid;
    out.g     = in.g;

#pragma HLS ARRAY_PARTITION variable=out.aggLR complete

    cost_t laneSum[PAR];
#pragma HLS ARRAY_PARTITION variable=laneSum complete


    out.newTBWord = 0;

ReduceLanes:
    for (int lane = 0; lane < PAR; ++lane)
    {
#pragma HLS UNROLL

        out.aggLR[lane] = in.aggLR[lane];

        out.newTBWord.range((lane + 1) * COST_BITS - 1, lane * COST_BITS)
        		= in.aggTB[lane];

        laneSum[lane] = sat12(wide_t(in.aggLR[lane]) + wide_t(in.aggTB[lane]));
    }

    cost_t minLR01 = (in.aggLR[1] < in.aggLR[0]) ? in.aggLR[1] : in.aggLR[0];

    cost_t minLR23 = (in.aggLR[3] < in.aggLR[2]) ? in.aggLR[3] : in.aggLR[2];

    out.groupMinLR = (minLR23 < minLR01) ? minLR23 : minLR01;

    cost_t minTB01 = (in.aggTB[1] < in.aggTB[0]) ? in.aggTB[1] : in.aggTB[0];

    cost_t minTB23 = (in.aggTB[3] < in.aggTB[2]) ? in.aggTB[3] : in.aggTB[2];

    out.groupMinTB = (minTB23 < minTB01) ? minTB23 : minTB01;

    cost_t bestCost01;
    disp_t bestDisp01;

    if (laneSum[1] < laneSum[0])
    {
        bestCost01 = laneSum[1];
        bestDisp01 = disp_t(in.g * PAR + 1);
    }
    else
    {
        bestCost01 = laneSum[0];
        bestDisp01 = disp_t(in.g * PAR);
    }

    cost_t bestCost23;
    disp_t bestDisp23;

    if (laneSum[3] < laneSum[2])
    {
        bestCost23 = laneSum[3];
        bestDisp23 = disp_t(in.g * PAR + 3);
    }
    else
    {
        bestCost23 = laneSum[2];
        bestDisp23 = disp_t(in.g * PAR + 2);
    }

    if (bestCost23 < bestCost01)
    {
        out.groupBestCost = bestCost23;
        out.groupBestDisp = bestDisp23;
    }
    else
    {
        out.groupBestCost = bestCost01;
        out.groupBestDisp = bestDisp01;
    }
}



static void commit_prev_costs(
    cost_t prevCostL[DISP],
    const cost_t aggLR_arr[DISP])
{
#pragma HLS INLINE off

CopyPrevLR:
    for (int d = 0; d < DISP; ++d)
    {
	#pragma HLS UNROLL
        prevCostL[d]    = aggLR_arr[d];
    }
}


////////////////////////////////////////////////////////////
static void prepare_stage(
    hls::stream<CostPacket>& coststream,
    hls::stream<PrepPacket>& prepstream,
    const cost_t prevCostL[DISP],
    const group_t prevCostT[GROUPS],
    cost_t minPrevLR,
    cost_t minPrevTB)
{
#pragma HLS INLINE off
	group_t tbPrevWord;
	for(int lane = 0; lane < PAR; ++lane)
	{
		tbPrevWord.range((lane + 1) * COST_BITS -1, lane * COST_BITS) = INF_COST;
	}

    group_t tbCurWord  = prevCostT[0];
    group_t tbNextWord = prevCostT[1];

PrepareGroups:
    for (int g = 0; g < GROUPS; ++g)
    {
#pragma HLS PIPELINE II=1

        CostPacket pkt = coststream.read();

        PrepPacket prep;

        prepare_group(
            pkt, g,
            prevCostL,
            tbPrevWord,
            tbCurWord,
            tbNextWord,
            minPrevLR,
            minPrevTB,
            prep);

        prepstream.write(prep);

        tbPrevWord = tbCurWord;
        tbCurWord  = tbNextWord;

        if (g + 2 < GROUPS)
            tbNextWord = prevCostT[g + 2];
        else
        {
        	for(int lane = 0; lane < PAR; ++lane)
        	{
        		tbNextWord.range((lane + 1) * COST_BITS -1, lane * COST_BITS) = INF_COST;
        	}
        }

    }
}


static void aggregate_stage(
    hls::stream<PrepPacket>& prepstream,
    hls::stream<AggPacket>& aggstream)
{
AggregateGroups:
    for (int g = 0; g < GROUPS; ++g)
    {
#pragma HLS PIPELINE II=1

        PrepPacket prep = prepstream.read();

        AggPacket agg;
        aggregate_group(prep, agg);

        aggstream.write(agg);
    }
}

static void reduce_stage(
    hls::stream<AggPacket>& aggstream,
    cost_t nextCostL_out[DISP],
    group_t nextCostT_out[GROUPS],
    cost_t& minPrevLR_out,
    cost_t& minPrevTB_out,
    disp_t& bestDisp,
    bool& validPixel_out)
{
    cost_t runMinLR = INF_COST;
    cost_t runMinTB = INF_COST;
    cost_t bestCost = INF_COST;

    validPixel_out = false;
    bestDisp = 0;

ReduceGroups:
    for (int g = 0; g < GROUPS; ++g)
    {
	#pragma HLS PIPELINE II=1

        AggPacket agg = aggstream.read();

        ResultPacket result;
        reduce_group(agg, result);

        if (result.valid)
        {
            validPixel_out = true;

            for (int lane = 0; lane < PAR; ++lane)
            {
			#pragma HLS UNROLL
                int d = g * PAR + lane;
                nextCostL_out[d] = result.aggLR[lane];
            }

            nextCostT_out[g] = result.newTBWord;

            if (result.groupMinLR < runMinLR)
                runMinLR = result.groupMinLR;

            if (result.groupMinTB < runMinTB)
                runMinTB = result.groupMinTB;

            if (result.groupBestCost < bestCost)
            {
                bestCost = result.groupBestCost;
                bestDisp = result.groupBestDisp;
            }
        }
    }

    if (validPixel_out)
    {
        minPrevLR_out = runMinLR;
        minPrevTB_out = runMinTB;
    }
}









/////////////////////////////////////////////////////////////


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
						const cost_t prevCostL_in[DISP],
						const group_t prevCostT_in[GROUPS],
						cost_t nextCostL_out[DISP],
						group_t nextCostT_out[GROUPS],
						cost_t minPrevLR_in,
						cost_t minPrevTB_in,
						cost_t& minPrevLR_out,
						cost_t& minPrevTB_out)
{
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<PrepPacket> prepstream;
    hls::stream<AggPacket>  aggstream;

#pragma HLS STREAM variable=prepstream depth=2
#pragma HLS STREAM variable=aggstream  depth=2

    disp_t bestDisp = 0;
    bool validPixel;

    prepare_stage(coststream,
				prepstream,
				prevCostL_in,
				prevCostT_in,
				minPrevLR_in,
				minPrevTB_in);

    aggregate_stage(prepstream, aggstream);

    reduce_stage(aggstream,
    			nextCostL_out,
				nextCostT_out,
				minPrevLR_out,
				minPrevTB_out,
				bestDisp,
				validPixel);

    return bestDisp;
}

static disp_t process_cost_groups(
    bool interior,
    census_t leftDesc,
    census_t rightCensusHistory[DISP],
    const cost_t prevCostL_in[DISP],
    const group_t prevCostT_in[GROUPS],
    cost_t nextCostL_out[DISP],
    group_t nextCostT_out[GROUPS],
    cost_t minPrevLR_in,
    cost_t minPrevTB_in,
    cost_t& minPrevLR_out,
    cost_t& minPrevTB_out,
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

    return col_backend(
        coststream,
        prevCostL_in,
        prevCostT_in,
        nextCostL_out,
        nextCostT_out,
        minPrevLR_in,
        minPrevTB_in,
        minPrevLR_out,
        minPrevTB_out);
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
    static cost_t costL_A[DISP];
    static cost_t costL_B[DISP];

    #pragma HLS ARRAY_PARTITION variable=costL_A complete
    #pragma HLS ARRAY_PARTITION variable=costL_B complete

    static group_t costT_A[IMG_W][GROUPS];
    static group_t costT_B[IMG_W][GROUPS];

    #pragma HLS bind_storage variable=costT_A type=RAM_2P impl=BRAM
    #pragma HLS bind_storage variable=costT_B type=RAM_2P impl=BRAM

    static cost_t aggTB_arr[DISP];
#pragma HLS ARRAY_PARTITION variable=aggTB_arr complete dim=1

    pix_t leftWin[WIN][WIN];
    pix_t rightStripe[WIN][RIGHT_STRIPE_W];

#pragma HLS ARRAY_PARTITION variable=leftWin complete dim=0
#pragma HLS ARRAY_PARTITION variable=rightStripe complete dim=1

    census_t rightCensusHistory[DISP];
#pragma HLS ARRAY_PARTITION variable=rightCensusHistory complete

    static cost_t minT_A[IMG_W];
    static cost_t minT_B[IMG_W];

    /* center offset */
    const int cx = WIN >> 1;
    const int cy = WIN >> 1;

    bool tbReadA = true;
Row:
    for (int r = 0; r < IMG_H; r++)
    {
    	int right_wr = RIGHT_STRIPE_W - 1;

    	cost_t minPrevLR = 0;

    	bool lrReadA = true;

    	bram_word_t disp_word = 0;

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
            costL_A[d] = cost_t(0);
            costL_B[d] = cost_t(0);
        }

        if(r == 0)
        {
        InitTBRow:
			for (int c = 0; c < IMG_W; ++c)
			{
			#pragma HLS LOOP_TRIPCOUNT min=IMG_W max=IMG_W
		        minT_A[c] = cost_t(0);
		        minT_B[c] = cost_t(0);

				InitTBRowD:
				for (int g = 0; g < GROUPS; ++g)
				{
				#pragma HLS UNROLL factor=2
		            costT_A[c][g] = group_t(0);
		            costT_B[c][g] = group_t(0);
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
    			disp_t outDisp;
    			cost_t newMinLR = minPrevLR;

    			if (tbReadA)
    			{
    			    if (lrReadA)
    			    {
    			        outDisp = process_cost_groups(
    			            interior,
    			            leftDesc,
    			            rightCensusHistory,
    			            costL_A,
    			            costT_A[out_c],
    			            costL_B,
    			            costT_B[out_c],
    			            minPrevLR,
    			            minT_A[out_c],
    			            newMinLR,
    			            minT_B[out_c],
    			            out_c);
    			    }
    			    else
    			    {
    			        // Previous column LR = B, current column LR = A
    			        outDisp = process_cost_groups(
    			            interior,
    			            leftDesc,
    			            rightCensusHistory,
    			            costL_B,
    			            costT_A[out_c],
    			            costL_A,
    			            costT_B[out_c],
    			            minPrevLR,
    			            minT_A[out_c],
    			            newMinLR,
    			            minT_B[out_c],
    			            out_c);
    			    }
    			}
    			else
    			{
    			    // Previous row is in T_B, current row goes to T_A

    			    if (lrReadA)
    			    {
    			        outDisp = process_cost_groups(
    			            interior,
    			            leftDesc,
    			            rightCensusHistory,
    			            costL_A,
    			            costT_B[out_c],
    			            costL_B,
    			            costT_A[out_c],
    			            minPrevLR,
    			            minT_B[out_c],
    			            newMinLR,
    			            minT_A[out_c],
    			            out_c);
    			    }
    			    else
    			    {
    			        outDisp = process_cost_groups(
    			            interior,
    			            leftDesc,
    			            rightCensusHistory,
    			            costL_B,
    			            costT_B[out_c],
    			            costL_A,
    			            costT_A[out_c],
    			            minPrevLR,
    			            minT_B[out_c],
    			            newMinLR,
    			            minT_A[out_c],
    			            out_c);
    			    }
    			}
    			if (interior)
    			{
    			    minPrevLR = newMinLR;
    			    lrReadA = !lrReadA;
    			}





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

    	if (r >= WIN - 1)
    	{
    	    tbReadA = !tbReadA;
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
