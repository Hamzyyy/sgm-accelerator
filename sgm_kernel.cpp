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

static inline void update_line_buffers(pix_t bufL[WIN][IMG_W],
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

static void update_sliding_windows(pix_t bufL[WIN][IMG_W],
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

					descriptor[0] = rightStripe[wy][physIndex]
									< centerR;
            }
        }
        return descriptor;
}

static void compute_census_cost_vector(census_t leftDesc,
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
				census_t diff = leftDesc ^
						rightCensusHistory[d];
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


struct PrepPacket
{
    bool valid;
    int g;
    cost_t minPrevLR;
    cost_t minPrevTB;
    cost_t minPrevTL;
    cost_t minPrevTR;
    cost_t curCost[PAR];
    cost_t lrCenter[PAR];
    cost_t lrLeft[PAR];
    cost_t lrRight[PAR];
    cost_t tbCenter[PAR];
    cost_t tbLeft[PAR];
    cost_t tbRight[PAR];
    cost_t tlCenter[PAR];
    cost_t tlLeft[PAR];
    cost_t tlRight[PAR];
    cost_t trCenter[PAR];
    cost_t trLeft[PAR];
    cost_t trRight[PAR];
};

static void prepare_group(const CostPacket& pkt,
								int g,
								const cost_t prevCostL[DISP],
								group_t tbPrevWord,
								group_t tbCurWord,
								group_t tbNextWord,
								group_t tlPrevWord,
								group_t tlCurWord,
								group_t tlNextWord,
								group_t trPrevWord,
								group_t trCurWord,
								group_t trNextWord,
								cost_t minPrevLR,
								cost_t minPrevTB,
								cost_t minPrevTL,
								cost_t minPrevTR,
								PrepPacket& out)
{
    out.valid = pkt.valid;
    out.g = g;

    out.minPrevLR = minPrevLR;
    out.minPrevTB = minPrevTB;
    out.minPrevTL = minPrevTL;
    out.minPrevTR = minPrevTR;

    cost_t tbCur[PAR];
    cost_t tlCur[PAR];
    cost_t trCur[PAR];

#pragma HLS ARRAY_PARTITION variable=tbCur complete
#pragma HLS ARRAY_PARTITION variable=tlCur complete
#pragma HLS ARRAY_PARTITION variable=trCur complete


UnpackCurrentWords:
    for(int lane = 0; lane < PAR; ++lane)
    {
#pragma HLS UNROLL

        tbCur[lane] = tbCurWord.range((lane + 1) *
        		COST_BITS - 1, lane * COST_BITS);

        tlCur[lane] = tlCurWord.range((lane + 1) *
        		COST_BITS - 1, lane * COST_BITS);

        trCur[lane] = trCurWord.range((lane + 1) *
        		COST_BITS - 1, lane * COST_BITS);
    }

    cost_t tbLeftBoundary = tbPrevWord.range( PAR *
    		COST_BITS - 1, (PAR - 1) * COST_BITS);

    cost_t tbRightBoundary = tbNextWord.range( COST_BITS - 1,
    		0);

    cost_t tlLeftBoundary = tlPrevWord.range(PAR * COST_BITS - 1,
    		(PAR - 1) * COST_BITS);

    cost_t tlRightBoundary = tlNextWord.range(COST_BITS - 1, 0);

    cost_t trLeftBoundary = trPrevWord.range(PAR * COST_BITS - 1,
            (PAR - 1) * COST_BITS);

    cost_t trRightBoundary = trNextWord.range(COST_BITS - 1, 0);

PrepLanes:
    for(int lane = 0; lane < PAR; ++lane)
    {
#pragma HLS UNROLL

        int d = g * PAR + lane;
        out.curCost[lane] = pkt.curCost[lane];

        /* LR */
        out.lrCenter[lane] = prevCostL[d];
        out.lrLeft[lane]   = (d > 0) ? prevCostL[d - 1] :
        		INF_COST;
        out.lrRight[lane]  = (d < DISP - 1) ? prevCostL[d + 1] :
        		INF_COST;

        /* TB */
        out.tbCenter[lane] = tbCur[lane];

        if(lane == 0)
            out.tbLeft[lane] = (d > 0) ? tbLeftBoundary : INF_COST;
        else
            out.tbLeft[lane] = (d > 0) ? tbCur[lane - 1] : INF_COST;

        if(lane == PAR - 1)
            out.tbRight[lane] = (d < DISP - 1) ? tbRightBoundary
            		: INF_COST;
        else
            out.tbRight[lane] = (d < DISP - 1) ? tbCur[lane + 1]
													   : INF_COST;

        /* TL */
        out.tlCenter[lane] = tlCur[lane];

        if(lane == 0)
            out.tlLeft[lane] = (d > 0) ? tlLeftBoundary
            		: INF_COST;
        else
            out.tlLeft[lane] = (d > 0) ? tlCur[lane - 1]
					: INF_COST;

        if(lane == PAR - 1)
            out.tlRight[lane] = (d < DISP - 1) ? tlRightBoundary
            		: INF_COST;
        else
            out.tlRight[lane] = (d < DISP - 1) ? tlCur[lane + 1]
					: INF_COST;

        /* TR */
        out.trCenter[lane] = trCur[lane];

        if(lane == 0)
            out.trLeft[lane] = (d > 0) ? trLeftBoundary : INF_COST;
        else
            out.trLeft[lane] = (d > 0) ? trCur[lane - 1] : INF_COST;

        if(lane == PAR - 1)
            out.trRight[lane] = (d < DISP - 1) ? trRightBoundary
            		: INF_COST;
        else
            out.trRight[lane] = (d < DISP - 1) ? trCur[lane + 1]
					: INF_COST;
    }
}



struct AggPacket
{
    bool valid;
    int g;
    cost_t aggLR[PAR];
    cost_t aggTB[PAR];
    cost_t aggTL[PAR];
    cost_t aggTR[PAR];
};



static void aggregate_group(const PrepPacket& in,
                            AggPacket& out)
{
    out.valid = in.valid;
    out.g = in.g;

LaneLoop:
    for(int lane = 0; lane < PAR; ++lane)
    {
#pragma HLS UNROLL

        /* ---------------- LR ---------------- */
        cost_t p0_LR = in.lrCenter[lane];
        cost_t p1_LR =
            sat12(wide_t(in.lrLeft[lane]) + wide_t(P1));
        cost_t p2_LR =
            sat12(wide_t(in.lrRight[lane]) + wide_t(P1));
        cost_t p3_LR =
            sat12(wide_t(in.minPrevLR) + wide_t(P2));

        cost_t minLR = p0_LR;
        if(p1_LR < minLR) minLR = p1_LR;
        if(p2_LR < minLR) minLR = p2_LR;
        if(p3_LR < minLR) minLR = p3_LR;

        if(in.curCost[lane] == INF_COST)
            out.aggLR[lane] = INF_COST;
        else
            out.aggLR[lane] =sat12(wide_t(in.curCost[lane]) +
                      wide_t(minLR) - wide_t(in.minPrevLR));


        /* ---------------- TB ---------------- */

        cost_t p0_TB = in.tbCenter[lane];
        cost_t p1_TB = sat12(wide_t(in.tbLeft[lane])
        		+ wide_t(P1));
        cost_t p2_TB = sat12(wide_t(in.tbRight[lane])
        		+ wide_t(P1));
        cost_t p3_TB = sat12(wide_t(in.minPrevTB)
        		+ wide_t(P2));

        cost_t minTB = p0_TB;
        if(p1_TB < minTB) minTB = p1_TB;
        if(p2_TB < minTB) minTB = p2_TB;
        if(p3_TB < minTB) minTB = p3_TB;

        if(in.curCost[lane] == INF_COST)
            out.aggTB[lane] = INF_COST;
        else
            out.aggTB[lane] = sat12(wide_t(in.curCost[lane])
            		+ wide_t(minTB) - wide_t(in.minPrevTB));


        /* ---------------- TL ---------------- */
        cost_t p0_TL = in.tlCenter[lane];
        cost_t p1_TL = sat12(wide_t(in.tlLeft[lane])
        		+ wide_t(P1));
        cost_t p2_TL = sat12(wide_t(in.tlRight[lane])
        		+ wide_t(P1));
        cost_t p3_TL = sat12(wide_t(in.minPrevTL) + wide_t(P2));

        cost_t minTL = p0_TL;
        if(p1_TL < minTL) minTL = p1_TL;
        if(p2_TL < minTL) minTL = p2_TL;
        if(p3_TL < minTL) minTL = p3_TL;

        if(in.curCost[lane] == INF_COST)
            out.aggTL[lane] = INF_COST;
        else
            out.aggTL[lane] = sat12(wide_t(in.curCost[lane])
            		+ wide_t(minTL) - wide_t(in.minPrevTL));


        /* ---------------- TR ---------------- */
        cost_t p0_TR = in.trCenter[lane];
        cost_t p1_TR = sat12(wide_t(in.trLeft[lane])
        		+ wide_t(P1));
        cost_t p2_TR = sat12(wide_t(in.trRight[lane])
        		+ wide_t(P1));
        cost_t p3_TR = sat12(wide_t(in.minPrevTR)
        		+ wide_t(P2));

        cost_t minTR = p0_TR;
        if(p1_TR < minTR) minTR = p1_TR;
        if(p2_TR < minTR) minTR = p2_TR;
        if(p3_TR < minTR) minTR = p3_TR;

        if(in.curCost[lane] == INF_COST)
            out.aggTR[lane] = INF_COST;
        else
            out.aggTR[lane] = sat12(wide_t(in.curCost[lane])
            		+ wide_t(minTR) - wide_t(in.minPrevTR));
    }
}



struct ResultPacket
{
    bool valid;
    int g;
    cost_t groupMinLR;
    cost_t groupMinTB;
    cost_t groupMinTL;
    cost_t groupMinTR;
    wide_t groupBestCost;
    disp_t groupBestDisp;
    group_t newTBWord;
    group_t newTLWord;
    group_t newTRWord;
    cost_t aggLR[PAR];
};




static void reduce_group(const AggPacket& in,
                         ResultPacket& out)
{
    out.valid = in.valid;
    out.g = in.g;

#pragma HLS ARRAY_PARTITION variable=out.aggLR complete

    wide_t laneSum[PAR];

    cost_t lrTree[PAR];
    cost_t tbTree[PAR];
    cost_t tlTree[PAR];
    cost_t trTree[PAR];

    wide_t costTree[PAR];
    disp_t dispTree[PAR];

#pragma HLS ARRAY_PARTITION variable=laneSum complete
#pragma HLS ARRAY_PARTITION variable=lrTree complete
#pragma HLS ARRAY_PARTITION variable=tbTree complete
#pragma HLS ARRAY_PARTITION variable=tlTree complete
#pragma HLS ARRAY_PARTITION variable=trTree complete
#pragma HLS ARRAY_PARTITION variable=costTree complete
#pragma HLS ARRAY_PARTITION variable=dispTree complete

    out.newTBWord = 0;
    out.newTLWord = 0;
    out.newTRWord = 0;


ReduceLanes:
    for(int lane = 0; lane < PAR; ++lane)
    {
#pragma HLS UNROLL

        out.aggLR[lane] = in.aggLR[lane];

        out.newTBWord.range((lane + 1) * COST_BITS - 1,
            lane * COST_BITS) = in.aggTB[lane];

        out.newTLWord.range((lane + 1) * COST_BITS - 1,
            lane * COST_BITS) = in.aggTL[lane];

        out.newTRWord.range((lane + 1) * COST_BITS - 1,
            lane * COST_BITS) = in.aggTR[lane];


        wide_t sum1 = wide_t(in.aggLR[lane]) +
            wide_t(in.aggTB[lane]);

        wide_t sum2 = wide_t(in.aggTL[lane]) +
            wide_t(in.aggTR[lane]);

        laneSum[lane] = sum1 + sum2;


        lrTree[lane] = in.aggLR[lane];
        tbTree[lane] = in.aggTB[lane];
        tlTree[lane] = in.aggTL[lane];
        trTree[lane] = in.aggTR[lane];

        costTree[lane] = laneSum[lane];
        dispTree[lane] =
            disp_t(in.g * PAR + lane);
    }

ReductionStages:
    for(int step = PAR / 2; step > 0; step >>= 1)
    {
#pragma HLS UNROLL

ReductionPairs:
        for(int i = 0; i < PAR; ++i)
        {
#pragma HLS UNROLL

            if(i < step)
            {
                int j = i + step;

                if(lrTree[j] < lrTree[i])
                    lrTree[i] = lrTree[j];

                if(tbTree[j] < tbTree[i])
                    tbTree[i] = tbTree[j];

                if(tlTree[j] < tlTree[i])
                    tlTree[i] = tlTree[j];

                if(trTree[j] < trTree[i])
                    trTree[i] = trTree[j];


                if(costTree[j] < costTree[i] ||
                  (costTree[j] == costTree[i] &&
                   dispTree[j] < dispTree[i]))
                {
                    costTree[i] = costTree[j];
                    dispTree[i] = dispTree[j];
                }
            }
        }
    }


    out.groupMinLR = lrTree[0];
    out.groupMinTB = tbTree[0];
    out.groupMinTL = tlTree[0];
    out.groupMinTR = trTree[0];

    out.groupBestCost = costTree[0];
    out.groupBestDisp = dispTree[0];
}


static void commit_prev_costs(cost_t prevCostL[DISP],
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


static void prepare_stage(hls::stream<CostPacket>& coststream,
						hls::stream<PrepPacket>& prepstream,
						const cost_t prevCostL[DISP],
						const group_t prevCostT[GROUPS],
						const group_t prevCostTL[GROUPS],
						const group_t prevCostTR[GROUPS],
						cost_t minPrevLR,
						cost_t minPrevTB,
						cost_t minPrevTL,
						cost_t minPrevTR)
{
#pragma HLS INLINE off

    group_t tbPrevWord = 0;
    group_t tlPrevWord = 0;
    group_t trPrevWord = 0;

InitPrevBoundary:
    for(int lane = 0; lane < PAR; ++lane)
    {
#pragma HLS UNROLL

        tbPrevWord.range((lane + 1) * COST_BITS - 1,
        		lane * COST_BITS) = INF_COST;

        tlPrevWord.range((lane + 1) * COST_BITS - 1,
            lane * COST_BITS) = INF_COST;

        trPrevWord.range((lane + 1) * COST_BITS - 1,
            lane * COST_BITS) = INF_COST;
    }


    group_t tbCurWord  = prevCostT[0];
    group_t tbNextWord = prevCostT[1];
    group_t tlCurWord  = prevCostTL[0];
    group_t tlNextWord = prevCostTL[1];
    group_t trCurWord  = prevCostTR[0];
    group_t trNextWord = prevCostTR[1];

PrepareGroups:
    for(int g = 0; g < GROUPS; ++g)
    {
#pragma HLS PIPELINE II=1

        CostPacket pkt = coststream.read();
        PrepPacket prep;

        prepare_group(pkt,
					g,
					prevCostL,
					tbPrevWord,
					tbCurWord,
					tbNextWord,
					tlPrevWord,
					tlCurWord,
					tlNextWord,
					trPrevWord,
					trCurWord,
					trNextWord,
					minPrevLR,
					minPrevTB,
					minPrevTL,
					minPrevTR,
					prep);

        prepstream.write(prep);

        tbPrevWord = tbCurWord;
        tbCurWord  = tbNextWord;
        tlPrevWord = tlCurWord;
        tlCurWord  = tlNextWord;
        trPrevWord = trCurWord;
        trCurWord  = trNextWord;

        if(g + 2 < GROUPS)
        {
            tbNextWord = prevCostT[g + 2];
            tlNextWord = prevCostTL[g + 2];
            trNextWord = prevCostTR[g + 2];
        }
        else
        {
        SetRightBoundary:
            for(int lane = 0; lane < PAR; ++lane)
            {
#pragma HLS UNROLL

                tbNextWord.range((lane + 1) * COST_BITS - 1,
                    lane * COST_BITS) = INF_COST;

                tlNextWord.range((lane + 1) * COST_BITS - 1,
                    lane * COST_BITS) = INF_COST;

                trNextWord.range((lane + 1) * COST_BITS - 1,
                    lane * COST_BITS) = INF_COST;
            }
        }
    }
}


static void aggregate_stage(hls::stream<PrepPacket>& prepstream,
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



static void reduce_stage(hls::stream<AggPacket>& aggstream,
						cost_t nextCostL_out[DISP],
						group_t nextCostT_out[GROUPS],
						group_t nextCostTL_out[GROUPS],
						group_t nextCostTR_out[GROUPS],
						cost_t& minPrevLR_out,
						cost_t& minPrevTB_out,
						cost_t& minPrevTL_out,
						cost_t& minPrevTR_out,
						disp_t& bestDisp,
						bool& validPixel_out)
{
    cost_t runMinLR = INF_COST;
    cost_t runMinTB = INF_COST;
    cost_t runMinTL = INF_COST;
    cost_t runMinTR = INF_COST;

    wide_t bestCost = 0;
    bool bestFound = false;

    validPixel_out = false;
    bestDisp = 0;


ReduceGroups:
    for(int g = 0; g < GROUPS; ++g)
    {
#pragma HLS PIPELINE II=1

        AggPacket agg = aggstream.read();

        ResultPacket result;
        reduce_group(agg, result);

        if(result.valid)
        {
            validPixel_out = true;

CopyLR:
            for(int lane = 0; lane < PAR; ++lane)
            {
#pragma HLS UNROLL

                int d = g * PAR + lane;
                nextCostL_out[d] = result.aggLR[lane];
            }

            nextCostT_out[g] = result.newTBWord;
            nextCostTL_out[g] = result.newTLWord;
            nextCostTR_out[g] = result.newTRWord;

            if(result.groupMinLR < runMinLR)
                runMinLR = result.groupMinLR;

            if(result.groupMinTB < runMinTB)
                runMinTB = result.groupMinTB;

            if(result.groupMinTL < runMinTL)
                runMinTL = result.groupMinTL;

            if(result.groupMinTR < runMinTR)
                runMinTR = result.groupMinTR;


            if(!bestFound ||
               result.groupBestCost < bestCost ||
              (result.groupBestCost == bestCost &&
               result.groupBestDisp < bestDisp))
            {
                bestCost = result.groupBestCost;
                bestDisp = result.groupBestDisp;
                bestFound = true;
            }
        }
    }


    if(validPixel_out)
    {
        minPrevLR_out = runMinLR;
        minPrevTB_out = runMinTB;
        minPrevTL_out = runMinTL;
        minPrevTR_out = runMinTR;
    }
}


static bool prepare_census_column(bram_word_t left[FRAME_WORDS],
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


static void col_frontend(hls::stream<CostPacket>& coststream,
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
	            compute_census_cost_vector(leftDesc,
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
						const group_t prevCostTL_in[GROUPS],
						const group_t prevCostTR_in[GROUPS],
						cost_t nextCostL_out[DISP],
						group_t nextCostT_out[GROUPS],
						group_t nextCostTL_out[GROUPS],
						group_t nextCostTR_out[GROUPS],
						cost_t minPrevLR_in,
						cost_t minPrevTB_in,
						cost_t minPrevTL_in,
						cost_t minPrevTR_in,
						cost_t& minPrevLR_out,
						cost_t& minPrevTB_out,
						cost_t& minPrevTL_out,
						cost_t& minPrevTR_out)
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
				prevCostTL_in,
				prevCostTR_in,
				minPrevLR_in,
				minPrevTB_in,
				minPrevTL_in,
				minPrevTR_in);

    aggregate_stage(prepstream, aggstream);

    reduce_stage(aggstream,
                 nextCostL_out,
                 nextCostT_out,
                 nextCostTL_out,
                 nextCostTR_out,
                 minPrevLR_out,
                 minPrevTB_out,
                 minPrevTL_out,
                 minPrevTR_out,

                 bestDisp,
                 validPixel);

    return bestDisp;
}

static void row_frontend(bram_word_t left[FRAME_WORDS],
						bram_word_t right[FRAME_WORDS],
						int r,
						pix_t bufL[WIN][IMG_W],
						pix_t bufR[WIN][IMG_W],
						pix_t leftWin[WIN][WIN],
						pix_t rightStripe[WIN][RIGHT_STRIPE_W],
						census_t rightCensusHistory[DISP],
						hls::stream<CostPacket>& coststream)
{
#pragma HLS INLINE off
	int right_wr = RIGHT_STRIPE_W - 1;
	census_t leftDesc;

	for (int d = 0; d < DISP; ++d)
	{
	#pragma HLS UNROLL factor=2
	    rightCensusHistory[d] = 0;
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
        bool interior = prepare_census_column(left,
											right,
											bufL,
											bufR,
											r,
											c,
											CENSUS_CX,
											leftWin,
											rightStripe,
											leftDesc,
											rightCensusHistory,
											right_wr);

        int out_c = c - CENSUS_CX;

        if (out_c >= 0)
        {
            for (int g = 0; g < GROUPS; ++g)
            {
			#pragma HLS PIPELINE II=1

                CostPacket pkt;
                pkt.valid = interior;

                if (interior)
                {
                    compute_census_cost_vector(leftDesc,
											rightCensusHistory,
											g,
											out_c,
											pkt.curCost);
                }
                coststream.write(pkt);
            }
        }
    }
}


static void row_backend(hls::stream<CostPacket>& coststream,
						bool rowValid,
						const group_t prevCostT[IMG_W][GROUPS],
						group_t nextCostT[IMG_W][GROUPS],
						const group_t prevCostTL[IMG_W][GROUPS],
						group_t nextCostTL[IMG_W][GROUPS],
						const group_t prevCostTR[IMG_W][GROUPS],
						group_t nextCostTR[IMG_W][GROUPS],
						const cost_t minPrevT[IMG_W],
						cost_t minNextT[IMG_W],
						const cost_t minPrevTL[IMG_W],
						cost_t minNextTL[IMG_W],
						const cost_t minPrevTR[IMG_W],
						cost_t minNextTR[IMG_W],
						hls::stream<disp_t>& dispstream)
{
#pragma HLS INLINE off

    cost_t costL_A[DISP];
    cost_t costL_B[DISP];

#pragma HLS ARRAY_PARTITION variable=costL_A complete
#pragma HLS ARRAY_PARTITION variable=costL_B complete

ResetLR:
    for (int d = 0; d < DISP; ++d)
    {
#pragma HLS UNROLL factor=2
        costL_A[d] = 0;
        costL_B[d] = 0;
    }

    cost_t minPrevLR = 0;
    bool lrReadA = true;

    group_t zeroPath[GROUPS];

    InitZeroPath:
        for (int g = 0; g < GROUPS; ++g)
        {
    #pragma HLS UNROLL factor=2
            zeroPath[g] = group_t(0);
        }


RowBackendCols:
    for (int out_c = 0; out_c < IMG_W - CENSUS_CX; ++out_c)
    {
        cost_t newMinLR = minPrevLR;
        cost_t newMinTB = minPrevT[out_c];
        cost_t newMinTL = (out_c > 0) ? minPrevTL[out_c - 1]
							: cost_t(0);
        cost_t newMinTR = minPrevTR[out_c + 1];

        disp_t outDisp;

        if(out_c == 0)
        {
        	if (lrReadA)
        	{
        		outDisp = col_backend(coststream,
        				costL_A,
						prevCostT[out_c],
						zeroPath,
						prevCostTR[out_c + 1],
						costL_B,
						nextCostT[out_c],
						nextCostTL[out_c],
						nextCostTR[out_c],
						minPrevLR,
						minPrevT[out_c],
						cost_t(0),
						minPrevTR[out_c + 1],
						newMinLR,
						newMinTB,
						newMinTL,
						newMinTR);
        	}
        	else
        	{
        		outDisp = col_backend(
        				coststream,
        	            costL_B,
        	            prevCostT[out_c],
        	            zeroPath,
        	            prevCostTR[out_c + 1],
        	            costL_A,
        	            nextCostT[out_c],
        	            nextCostTL[out_c],
        	            nextCostTR[out_c],
        	            minPrevLR,
        	            minPrevT[out_c],
        	            cost_t(0),
        	            minPrevTR[out_c + 1],
        	            newMinLR,
        	            newMinTB,
        	            newMinTL,
						newMinTR);
        	}
        }
        else
        {
        	if (lrReadA)
        	{
        		outDisp = col_backend(coststream,
									costL_A,
									prevCostT[out_c],
									prevCostTL[out_c - 1],
									prevCostTR[out_c + 1],
									costL_B,
									nextCostT[out_c],
									nextCostTL[out_c],
									nextCostTR[out_c],
									minPrevLR,
									minPrevT[out_c],
									minPrevTL[out_c - 1],
									minPrevTR[out_c + 1],
									newMinLR,
									newMinTB,
									newMinTL,
									newMinTR);
            }
            else
            {
                outDisp = col_backend(coststream,
									costL_B,
									prevCostT[out_c],
									prevCostTL[out_c - 1],
									prevCostTR[out_c + 1],
									costL_A,
									nextCostT[out_c],
									nextCostTL[out_c],
									nextCostTR[out_c],
									minPrevLR,
									minPrevT[out_c],
									minPrevTL[out_c - 1],
									minPrevTR[out_c + 1],
									newMinLR,
									newMinTB,
									newMinTL,
									newMinTR);
            }
        }
        if (rowValid)
        {
            minPrevLR = newMinLR;

            minNextT[out_c]  = newMinTB;
            minNextTL[out_c] = newMinTL;
            minNextTR[out_c] = newMinTR;

            lrReadA = !lrReadA;
        }

        dispstream.write(outDisp);
    }
}


static void row_output(hls::stream<disp_t>& dispstream,
					bram_word_t disp[FRAME_WORDS],
					int out_r,
					bool outputValid)
{
#pragma HLS INLINE off

    bram_word_t disp_word = 0;

RowOutputCols:
    for (int out_c = 0; out_c < IMG_W - CENSUS_CX; ++out_c)
    {
#pragma HLS PIPELINE II=1

        disp_t outDisp = dispstream.read();

        if (outputValid)
        {
            int pixel_idx = out_r * IMG_W + out_c;
            int word_idx  = pixel_idx >> 2;
            int byte_idx  = pixel_idx & 3;

            disp_word.range(byte_idx * 8 + 7,
							byte_idx * 8) = outDisp;

            if (byte_idx == 3)
            {
                disp[word_idx] = disp_word;
                disp_word = 0;
            }
        }
    }
    if (outputValid)
    {
        int last_pixel_idx = out_r * IMG_W + (IMG_W - 1);
        int last_word_idx  = last_pixel_idx >> 2;

        disp_word.range(31, 24) = 0;
        disp[last_word_idx] = disp_word;
    }
}

static void process_row(bram_word_t left[FRAME_WORDS],
						bram_word_t right[FRAME_WORDS],
						bram_word_t disp[FRAME_WORDS],
						int r,
						int out_r,
						pix_t bufL[WIN][IMG_W],
						pix_t bufR[WIN][IMG_W],
						pix_t leftWin[WIN][WIN],
						pix_t rightStripe[WIN][RIGHT_STRIPE_W],
						census_t rightCensusHistory[DISP],
						const group_t prevCostT[IMG_W][GROUPS],
						group_t nextCostT[IMG_W][GROUPS],
						const group_t prevCostTL[IMG_W][GROUPS],
						group_t nextCostTL[IMG_W][GROUPS],
						const group_t prevCostTR[IMG_W][GROUPS],
						group_t nextCostTR[IMG_W][GROUPS],
						const cost_t minPrevT[IMG_W],
						cost_t minNextT[IMG_W],
						const cost_t minPrevTL[IMG_W],
						cost_t minNextTL[IMG_W],
						const cost_t minPrevTR[IMG_W],
						cost_t minNextTR[IMG_W])
{
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<CostPacket> coststream;
    hls::stream<disp_t>     dispstream;

#pragma HLS STREAM variable=coststream depth=16
#pragma HLS STREAM variable=dispstream depth=4

    bool rowValid = (r >= WIN - 1);
    bool outputValid = (out_r >= 0);

    row_frontend(left,
				right,
				r,
				bufL,
				bufR,
				leftWin,
				rightStripe,
				rightCensusHistory,
				coststream);

    row_backend(coststream,
				rowValid,
				prevCostT,
				nextCostT,
				prevCostTL,
				nextCostTL,
				prevCostTR,
				nextCostTR,
				minPrevT,
				minNextT,
				minPrevTL,
				minNextTL,
				minPrevTR,
				minNextTR,
				dispstream);

    row_output(
        dispstream,
        disp,
        out_r,
        outputValid);
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
    static group_t costT_A[IMG_W][GROUPS];
    static group_t costT_B[IMG_W][GROUPS];

    static group_t costTL_A[IMG_W][GROUPS];
    static group_t costTL_B[IMG_W][GROUPS];

    static group_t costTR_A[IMG_W][GROUPS];
    static group_t costTR_B[IMG_W][GROUPS];

#pragma HLS bind_storage variable=costT_A type=RAM_2P impl=BRAM
#pragma HLS bind_storage variable=costT_B type=RAM_2P impl=BRAM

#pragma HLS bind_storage variable=costTL_A type=RAM_2P impl=BRAM
#pragma HLS bind_storage variable=costTL_B type=RAM_2P impl=BRAM

#pragma HLS bind_storage variable=costTR_A type=RAM_2P impl=BRAM
#pragma HLS bind_storage variable=costTR_B type=RAM_2P impl=BRAM



    pix_t leftWin[WIN][WIN];
    pix_t rightStripe[WIN][RIGHT_STRIPE_W];

#pragma HLS ARRAY_PARTITION variable=leftWin complete dim=0
#pragma HLS ARRAY_PARTITION variable=rightStripe complete dim=1

    census_t rightCensusHistory[DISP];
#pragma HLS ARRAY_PARTITION variable=rightCensusHistory complete

    static cost_t minT_A[IMG_W];
    static cost_t minT_B[IMG_W];

    static cost_t minTL_A[IMG_W];
    static cost_t minTL_B[IMG_W];

    static cost_t minTR_A[IMG_W];
    static cost_t minTR_B[IMG_W];

    //--------------------------------------------------
    // Initialize TB state once
    //--------------------------------------------------
InitPathRow:
    for (int c = 0; c < IMG_W; ++c)
    {
#pragma HLS LOOP_TRIPCOUNT min=IMG_W max=IMG_W

        minT_A[c] = cost_t(0);
        minT_B[c] = cost_t(0);

        minTL_A[c] = cost_t(0);
        minTL_B[c] = cost_t(0);

        minTR_A[c] = cost_t(0);
        minTR_B[c] = cost_t(0);

    InitPathRowD:
        for (int g = 0; g < GROUPS; ++g)
        {
#pragma HLS UNROLL factor=2
            costT_A[c][g] = group_t(0);
            costT_B[c][g] = group_t(0);

            costTL_A[c][g] = group_t(0);
            costTL_B[c][g] = group_t(0);

            costTR_A[c][g] = group_t(0);
            costTR_B[c][g] = group_t(0);
        }
    }


    //--------------------------------------------------
    // Row processing
    //--------------------------------------------------
    const int cy = WIN >> 1;

    bool tbReadA = true;

Row:
    for (int r = 0; r < IMG_H; ++r)
    {
        int out_r = r - cy;

        if (tbReadA)
        {
            // previous row = A
            // current row  = B
            process_row(left,
						right,
						disp,
						r,
						out_r,
						bufL,
						bufR,
						leftWin,
						rightStripe,
						rightCensusHistory,
						costT_A,
						costT_B,
						costTL_A,
						costTL_B,
						costTR_A,
						costTR_B,
						minT_A,
						minT_B,
						minTL_A,
						minTL_B,
						minTR_A,
						minTR_B);
        }
        else
        {
            // previous row = B
            // current row  = A
            process_row(left,
						right,
						disp,
						r,
						out_r,
						bufL,
						bufR,
						leftWin,
						rightStripe,
						rightCensusHistory,
						costT_B,
						costT_A,
						costTL_B,
						costTL_A,
						costTR_B,
						costTR_A,
						minT_B,
						minT_A,
						minTL_B,
						minTL_A,
						minTR_B,
						minTR_A);
        }

        if (r >= WIN - 1)
        {
            tbReadA = !tbReadA;
        }
    }
}
