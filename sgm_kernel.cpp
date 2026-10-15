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

    /* Temporary arrays for balanced reduction tree */
    cost_t lrTree[PAR];
    cost_t tbTree[PAR];
    cost_t costTree[PAR];
    disp_t dispTree[PAR];

#pragma HLS ARRAY_PARTITION variable=lrTree   complete
#pragma HLS ARRAY_PARTITION variable=tbTree   complete
#pragma HLS ARRAY_PARTITION variable=costTree complete
#pragma HLS ARRAY_PARTITION variable=dispTree complete

    out.newTBWord = 0;

ReduceLanes:
    for (int lane = 0; lane < PAR; ++lane)
    {
#pragma HLS UNROLL

        out.aggLR[lane] = in.aggLR[lane];

        out.newTBWord.range((lane + 1) * COST_BITS - 1, lane * COST_BITS)
        		= in.aggTB[lane];

        laneSum[lane] = sat12(wide_t(in.aggLR[lane]) + wide_t(in.aggTB[lane]));

        lrTree[lane]   = in.aggLR[lane];
        tbTree[lane]   = in.aggTB[lane];

        costTree[lane] = laneSum[lane];
        dispTree[lane] = disp_t(in.g * PAR + lane);
    }

    ReductionStages:
    for (int step = PAR / 2; step > 0; step >>= 1)
    {
	#pragma HLS UNROLL

    ReductionPairs:
        for (int i = 0; i < PAR; ++i)
        {
		#pragma HLS UNROLL

            if (i < step)
            {
                int j = i + step;

                if (lrTree[j] < lrTree[i])
                {
                    lrTree[i] = lrTree[j];
                }

                if (tbTree[j] < tbTree[i])
                {
                    tbTree[i] = tbTree[j];
                }

                if (costTree[j] < costTree[i])
                {
                    costTree[i] = costTree[j];
                    dispTree[i] = dispTree[j];
                }
            }
        }
    }
    out.groupMinLR = lrTree[0];
    out.groupMinTB = tbTree[0];

    out.groupBestCost = costTree[0];
    out.groupBestDisp = dispTree[0];
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

        prepare_group(pkt, g,
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

static void reduce_stage(hls::stream<AggPacket>& aggstream,
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


static void row_backend(
    hls::stream<CostPacket>& coststream,

    bool rowValid,

    const group_t prevCostT[IMG_W][GROUPS],
    group_t nextCostT[IMG_W][GROUPS],

    const cost_t minPrevT[IMG_W],
    cost_t minNextT[IMG_W],

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


RowBackendCols:
    for (int out_c = 0;
         out_c < IMG_W - CENSUS_CX;
         ++out_c)
    {
        cost_t newMinLR = minPrevLR;
        cost_t newMinTB = minPrevT[out_c];

        disp_t outDisp;

        if (lrReadA)
        {
            outDisp = col_backend(
                coststream,

                costL_A,
                prevCostT[out_c],

                costL_B,
                nextCostT[out_c],

                minPrevLR,
                minPrevT[out_c],

                newMinLR,
                newMinTB);
        }
        else
        {
            outDisp = col_backend(
                coststream,

                costL_B,
                prevCostT[out_c],

                costL_A,
                nextCostT[out_c],

                minPrevLR,
                minPrevT[out_c],

                newMinLR,
                newMinTB);
        }

        if (rowValid)
        {
            minPrevLR = newMinLR;
            minNextT[out_c] = newMinTB;

            lrReadA = !lrReadA;
        }

        dispstream.write(outDisp);
    }
}


static void row_output(
    hls::stream<disp_t>& dispstream,
    bram_word_t disp[FRAME_WORDS],
    int out_r,
    bool outputValid)
{
#pragma HLS INLINE off

    bram_word_t disp_word = 0;

RowOutputCols:
    for (int out_c = 0;
         out_c < IMG_W - CENSUS_CX;
         ++out_c)
    {
#pragma HLS PIPELINE II=1

        disp_t outDisp = dispstream.read();

        if (outputValid)
        {
            int pixel_idx = out_r * IMG_W + out_c;
            int word_idx  = pixel_idx >> 2;
            int byte_idx  = pixel_idx & 3;

            disp_word.range(
                byte_idx * 8 + 7,
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

static void process_row(
    bram_word_t left[FRAME_WORDS],
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

    const cost_t minPrevT[IMG_W],
    cost_t minNextT[IMG_W])
{
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<CostPacket> coststream;
    hls::stream<disp_t>     dispstream;

#pragma HLS STREAM variable=coststream depth=16
#pragma HLS STREAM variable=dispstream depth=4

    bool rowValid = (r >= WIN - 1);
    bool outputValid = (out_r >= 0);

    row_frontend(
        left,
        right,
        r,
        bufL,
        bufR,
        leftWin,
        rightStripe,
        rightCensusHistory,
        coststream);

    row_backend(
        coststream,
        rowValid,
        prevCostT,
        nextCostT,
        minPrevT,
        minNextT,
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

    #pragma HLS bind_storage variable=costT_A type=RAM_2P impl=BRAM
    #pragma HLS bind_storage variable=costT_B type=RAM_2P impl=BRAM

    pix_t leftWin[WIN][WIN];
    pix_t rightStripe[WIN][RIGHT_STRIPE_W];

#pragma HLS ARRAY_PARTITION variable=leftWin complete dim=0
#pragma HLS ARRAY_PARTITION variable=rightStripe complete dim=1

    census_t rightCensusHistory[DISP];
#pragma HLS ARRAY_PARTITION variable=rightCensusHistory complete

    static cost_t minT_A[IMG_W];
    static cost_t minT_B[IMG_W];

    //--------------------------------------------------
    // Initialize TB state once
    //--------------------------------------------------
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
            process_row(
                left,
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

                minT_A,
                minT_B);
        }
        else
        {
            // previous row = B
            // current row  = A
            process_row(
                left,
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

                minT_B,
                minT_A);
        }

        if (r >= WIN - 1)
        {
            tbReadA = !tbReadA;
        }
    }
}
