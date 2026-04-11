#include "sgm_params.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

/* Penalties */
const cost_t P1 = cost_t(10);
const cost_t P2 = cost_t(150);

static const cost_t INF_COST = cost_t(4095);

/* --------------------------------------------------------- */
/* Helper Function                                           */
/* --------------------------------------------------------- */
static inline void update_line_buffers(
		xf::cv::LineBuffer<WIN, IMG_W, pix_t>& bufL,
		xf::cv::LineBuffer<WIN, IMG_W, pix_t>& bufR,
		int c,
		pix_t pL,
		pix_t pR)
{
#pragma HLS INLINE
    bufL.shift_up(c);
    bufL.insert_bottom(pL, c);

    bufR.shift_up(c);
    bufR.insert_bottom(pR, c);
}

static inline void update_windows(
    xf::cv::LineBuffer<WIN, IMG_W, pix_t>& bufL,
    xf::cv::LineBuffer<WIN, IMG_W, pix_t>& bufR,
    xf::cv::Window<WIN, WIN, pix_t>& winL,
    xf::cv::Window<WIN, WIN, pix_t>& winR,
    int c,
    pix_t pL,
    pix_t pR)
{
#pragma HLS INLINE

    winL.shift_pixels_left();
    winR.shift_pixels_left();

WindowFill:
    for (int wy = 0; wy < WIN - 1; ++wy)
    {
	#pragma HLS UNROLL factor=2
        winL.insert_pixel(bufL.getval(wy, c), wy, WIN - 1);
        winR.insert_pixel(bufR.getval(wy, c), wy, WIN - 1);
    }

    winL.insert_pixel(pL, WIN - 1, WIN - 1);
    winR.insert_pixel(pR, WIN - 1, WIN - 1);
}

static inline pix_t safe_get(
		xf::cv::LineBuffer<WIN, IMG_W, pix_t>& buf,
		int wy,
		int col
		)
{
#pragma HLS INLINE
    if (col >= 0 && col < IMG_W)
        return buf.getval(wy, col);
    return pix_t(0);
}

static void compute_sad_cost_vector(
	    xf::cv::LineBuffer<WIN, IMG_W, pix_t>& bufL,
	    xf::cv::LineBuffer<WIN, IMG_W, pix_t>& bufR,
	    int c,
	    int cx,
	    cost_t curCost[DISP])
{
#pragma HLS INLINE off
	SAD_Loop:
	    for (int d = 0; d < DISP; d++)
	    {
	#pragma HLS UNROLL factor=2
	        cost_t sum = 0;

	    WinY:
	        for (int wy = 0; wy < WIN; wy++)
	        {
	#pragma HLS UNROLL factor=2
	        WinX:
	            for (int wx = 0; wx < WIN; wx++)
	            {
	#pragma HLS UNROLL factor=2
	                int colL  = c + wx - cx;
	                int colR = c - d + wx - cx;

	                pix_t lpx = safe_get(bufL, wy, colL);
	                pix_t rpx = safe_get(bufR, wy, colR);

	                sum += absdiff(lpx, rpx);
	            }
	        }
	        curCost[d] = sum;
	    }
}

using census_t = ap_uint<(WIN * WIN - 1)>;

static inline census_t make_census_desc(
    xf::cv::Window<WIN, WIN, pix_t>& win,
    int cx,
    int cy)
{
#pragma HLS INLINE

    census_t desc = 0;
    pix_t center_val = win.getval(cy, cx);
    int bit_idx = 0;

DescY:
    for (int wy = 0; wy < WIN; ++wy)
    {
    DescX:
        for (int wx = 0; wx < WIN; ++wx)
        {
            if (wy == cy && wx == cx)
                continue;

            pix_t px = win.getval(wy, wx);
            desc[bit_idx] = (px < center_val) ? 1 : 0;
            bit_idx++;
        }
    }

    return desc;
}

static inline cost_t hamming_distance(census_t a, census_t b)
{
#pragma HLS INLINE

    census_t x = a ^ b;
    cost_t cnt = 0;

HamLoop:
    for (int i = 0; i < WIN * WIN - 1; ++i)
    {
#pragma HLS UNROLL factor=2
        cnt += x[i];
    }

    return cnt;
}

static void compute_census_cost_vector(
    xf::cv::Window<WIN, WIN, pix_t>& winL,
    xf::cv::Window<WIN, WIN, pix_t>& winR,
    int r,
    int c,
    int cx,
    int cy,
    cost_t curCost[DISP])
{
#pragma HLS INLINE off

    census_t leftDesc = make_census_desc(winL, cx, cy);
    census_t rightDesc = make_census_desc(winR, cx, cy);
    cost_t ham0 = hamming_distance(leftDesc, rightDesc);

#ifndef __SYNTHESIS__
        if (r == 96 && c == 160)
        {
            std::cout << "LEFT window at (" << r << "," << c << ")\n";
            std::cout << "left center = " << int(winL.getval(cy, cx)) << "\n";

            for (int wy = 0; wy < WIN; ++wy)
            {
                for (int wx = 0; wx < WIN; ++wx)
                {
                    std::cout << int(winL.getval(wy, wx)) << " ";
                }
                std::cout << "\n";
            }

            std::cout << "leftDesc = " << leftDesc.to_uint() << "\n";
            std::cout << "RIGHT window\n";
            std::cout << "right center = " << int(winR.getval(cy, cx)) << "\n";

            for (int wy = 0; wy < WIN; ++wy)
            {
                for (int wx = 0; wx < WIN; ++wx)
                {
                    std::cout << int(winR.getval(wy, wx)) << " ";
                }
                std::cout << "\n";
            }

            std::cout << "rightDesc = " << rightDesc.to_uint() << "\n";
            std::cout << "ham0 = " << ham0 << "\n";
        }
#endif
CensusLoop:
    for (int d = 0; d < DISP; ++d)
    {
#pragma HLS UNROLL factor=2
    	curCost[d] = (d == 0) ? ham0 : cost_t(255);
    }
}

static cost_t reduce_min_vec(const cost_t vec[DISP])
{
#pragma HLS INLINE off
    cost_t minVal = INF_COST;

MinLoop:
    for (int d = 0; d < DISP; d++)
    {
	#pragma HLS UNROLL factor=2
        if (vec[d] < minVal)
            minVal = vec[d];
    }
    return minVal;
}

static disp_t aggregate_paths_and_select(
    const cost_t curCost[DISP],
    const cost_t prevCostL[DISP],
    const cost_t prevCostT_col[DISP],
    cost_t minPrevLR,
    cost_t minPrevTB,
    cost_t aggLR_arr[DISP],
    cost_t aggTB_arr[DISP],
    cost_t aggCost[DISP])
{
#pragma HLS INLINE off

    cost_t bestCost = INF_COST;
    disp_t bestDisp = 0;

AggregationLoop:
    for (int d = 0; d < DISP; d++)
    {
#pragma HLS UNROLL factor=2
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

        cost_t sum2 = sat12(aggLR + aggTB);
        aggCost[d] = sum2;

        if (sum2 < bestCost)
        {
            bestCost = sum2;
            bestDisp = disp_t(d);
        }
    }
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
#pragma HLS UNROLL factor=2
        prevCostL[d]    = aggLR_arr[d];
        prevCostT_col[d] = aggTB_arr[d];
    }
}

static void debug_linebuffer_rows(
    xf::cv::LineBuffer<WIN, IMG_W, pix_t>& buf,
    int r,
    int c,
    int cx)
{
#ifndef __SYNTHESIS__
    std::cout << "LineBuffer debug at (" << r << "," << c << ")\n";
#endif
    for (int wy = 0; wy < WIN; ++wy)
    {
#ifndef __SYNTHESIS__
        std::cout << "buf row " << wy << ": ";
#endif
        for (int wx = 0; wx < WIN; ++wx)
        {
            int col = c + wx - cx;
            pix_t v = 0;
            if (col >= 0 && col < IMG_W)
                v = buf.getval(wy, col);
#ifndef __SYNTHESIS__
            std::cout << int(v) << " ";
#endif
        }
#ifndef __SYNTHESIS__
        std::cout << "\n";
#endif
    }
}

/* --------------------------------------------------------- */
/* Top kernel                                                */
/* --------------------------------------------------------- */

void sgm_kernel(hls::stream<pix_t>& left,
                hls::stream<pix_t>& right,
                hls::stream<pix_t>& disp)
{
#pragma HLS INTERFACE axis         port=left   register
#pragma HLS INTERFACE axis         port=right  register
#pragma HLS INTERFACE axis         port=disp   register
#pragma HLS INTERFACE ap_ctrl_none port=return
//#pragma HLS DATAFLOW

    /* Line buffers for the left & right images */
    xf::cv::LineBuffer<WIN, IMG_W, pix_t> bufL;
    xf::cv::LineBuffer<WIN, IMG_W, pix_t> bufR;

    InitBuf:
    for (int wy = 0; wy < WIN; ++wy)
    {
        for (int c = 0; c < IMG_W; ++c)
        {
            bufL.val[wy][c] = 0;
            bufR.val[wy][c] = 0;
        }
    }

    xf::cv::Window<WIN, WIN, pix_t> winL;
    xf::cv::Window<WIN, WIN, pix_t> winR;

    InitWin:
    for (int wy = 0; wy < WIN; ++wy)
    {
        for (int wx = 0; wx < WIN; ++wx)
        {
            winL.val[wy][wx] = 0;
            winR.val[wy][wx] = 0;
        }
    }
    /* Cost arrays */
    static cost_t prevCostL[DISP];
#pragma HLS ARRAY_PARTITION variable=prevCostL complete dim=1

    static cost_t prevCostT[IMG_W][DISP];
#pragma HLS bind_storage variable=prevCostT type=RAM_1P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=prevCostT complete dim=2

    static cost_t curCost[DISP];
#pragma HLS ARRAY_PARTITION variable=curCost complete dim=1

    static cost_t aggCost[DISP];
#pragma HLS ARRAY_PARTITION variable=aggCost complete dim=1

    	static cost_t aggLR_arr[DISP];
        static cost_t aggTB_arr[DISP];
    #pragma HLS ARRAY_PARTITION variable=aggLR_arr complete dim=1
    #pragma HLS ARRAY_PARTITION variable=aggTB_arr complete dim=1

    /* center offsets */
    const int cy = WIN >> 1;
    const int cx = WIN >> 1;

Row:
    for (int r = 0; r < IMG_H; r++)
    {
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
				InitTBRowD:
				for (int d = 0; d < DISP; ++d)
				{
				#pragma HLS UNROLL factor=2
					prevCostT[c][d] = cost_t(0);
				}
			}
        }

    Col:
        for (int c = 0; c < IMG_W; c++)
        {
		//#pragma HLS PIPELINE II=2
		#pragma HLS DEPENDENCE variable=bufL inter false
		#pragma HLS DEPENDENCE variable=bufR inter false

            /* Stream next pixels */
            pix_t pL = left.read();
            pix_t pR = right.read();

            update_line_buffers(bufL, bufR, c, pL, pR);
            update_windows(bufL, bufR, winL, winR, c, pL, pR);
#ifndef __SYNTHESIS__
            if (r == 96 && c == 160)
            {
                std::cout << "Current input pL = " << int(pL)
                          << " pR = " << int(pR) << "\n";

                std::cout << "--- bufL ---\n";
                debug_linebuffer_rows(bufL, r, c, cx);

                std::cout << "--- bufR ---\n";
                debug_linebuffer_rows(bufR, r, c, cx);

                std::cout << "--- winL ---\n";
                for (int wy = 0; wy < WIN; ++wy)
                {
                    for (int wx = 0; wx < WIN; ++wx)
                    {
                        std::cout << int(winL.getval(wy, wx)) << " ";
                    }
                    std::cout << "\n";
                }

                std::cout << "--- winR ---\n";
                for (int wy = 0; wy < WIN; ++wy)
                {
                    for (int wx = 0; wx < WIN; ++wx)
                    {
                        std::cout << int(winR.getval(wy, wx)) << " ";
                    }
                    std::cout << "\n";
                }
            }
#endif

            /* Default output */
            pix_t outDisp = 0;

            if (r >= WIN - 1 && c >= DISP - 1)
            {
            	compute_sad_cost_vector(bufL, bufR, c, cx, curCost);

            	//compute_census_cost_vector(winL, winR, r, c, cx, cy, curCost);

                cost_t minPrevLR = reduce_min_vec(prevCostL);
                cost_t minPrevTB = reduce_min_vec(prevCostT[c]);

                disp_t bestDisp = aggregate_paths_and_select(
                    curCost,
                    prevCostL,
                    prevCostT[c],
                    minPrevLR,
                    minPrevTB,
                    aggLR_arr,
                    aggTB_arr,
                    aggCost);
#ifndef __SYNTHESIS__
                if (r == 96 && c == 160)
                {
                    std::cout << "DEBUG pixel (" << r << "," << c << ")\n";

                    for (int d = 0; d < 8; ++d)
                    {
                        std::cout << "curCost[" << d << "] = " << curCost[d] << "\n";
                    }

                    std::cout << "minPrevLR = " << minPrevLR << "\n";
                    std::cout << "minPrevTB = " << minPrevTB << "\n";

                    for (int d = 0; d < 8; ++d)
                    {
                        std::cout << "aggLR[" << d << "] = " << aggLR_arr[d]
                                  << " aggTB[" << d << "] = " << aggTB_arr[d]
                                  << " aggCost[" << d << "] = " << aggCost[d] << "\n";
                    }

                    std::cout << "bestDisp = " << int(bestDisp) << "\n";
                }
#endif
                commit_prev_costs(
                    prevCostL,
                    prevCostT[c],
                    aggLR_arr,
                    aggTB_arr);

                if (c >= DISP - 1)
                {
                    outDisp = bestDisp;
                }
            }

            disp.write(outDisp);
        }
    }
}
