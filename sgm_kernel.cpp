#include "sgm_params.hpp"

/* Penalties */
const cost_t P1 = cost_t(10);
const cost_t P2 = cost_t(150);

static const cost_t INF_COST = cost_t(4095);

void sgm_kernel(hls::stream<pix_t>& left,
                hls::stream<pix_t>& right,
                hls::stream<pix_t>& disp)
{
#pragma HLS INTERFACE axis         port=left   register
#pragma HLS INTERFACE axis         port=right  register
#pragma HLS INTERFACE axis         port=disp   register
#pragma HLS INTERFACE ap_ctrl_none port=return

    /* Line buffer for the left image window */
    xf::cv::LineBuffer<WIN, IMG_W, pix_t> bufL;

    /* Right image window buffer */
    static pix_t bufR[WIN][IMG_W];
#pragma HLS bind_storage variable=bufR type=RAM_2P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=bufR complete dim=1

    /* Cost arrays */
    static cost_t prevCostL[DISP];
#pragma HLS bind_storage variable=prevCostL type=RAM_1P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=prevCostL complete dim=1

    static cost_t curCost[DISP];
#pragma HLS bind_storage variable=curCost type=RAM_1P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=curCost complete dim=1

    static cost_t aggCost[DISP];
#pragma HLS bind_storage variable=aggCost type=RAM_1P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=aggCost complete dim=1

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
            prevCostL[d] = cost_t(0);
        }

    Col:
        for (int c = 0; c < IMG_W; c++)
        {
		//#pragma HLS PIPELINE II=10
		#pragma HLS DEPENDENCE variable=bufL inter false
		#pragma HLS DEPENDENCE variable=bufR inter false

            /* Stream next pixels */
            pix_t pL = left.read();
            pix_t pR = right.read();

            /* Update left line buffer at column c */
            bufL.shift_up(c);
            bufL.insert_bottom(pL, c);

        ShiftRows:
            for (int i = 0; i < WIN - 1; i++)
            {
            #pragma HLS UNROLL
                bufR[i][c] = bufR[i+1][c];
            }
            bufR[WIN - 1][c] = pR;

            /* Default output */
            pix_t outDisp = 0;

            if (r >= WIN - 1)
            {
                /* Calculate SAD matching cost for all disparities */
            SAD_Loop:
                for (int d = 0; d < DISP; d++)
                {
                    cost_t sum = 0;
                WinY:
                    for (int wy = 0; wy < WIN; wy++)
                    {
                    #pragma HLS UNROLL
                    WinX:
                        for (int wx = 0; wx < WIN; wx++)
                        {
                        #pragma HLS UNROLL
                            int colL = c + wx - cx;
                            pix_t lpx = pix_t(0);
                            if (colL >= 0 && colL < IMG_W)
                            {
                                lpx = bufL.getval(wy, colL);
                            }
                            int col_r = c - d + wx - cx;
                            pix_t rpx = pix_t(0);
                            if (col_r >= 0 && col_r < IMG_W)
                            {
                                rpx = bufR[wy][col_r];
                            }
                            sum += absdiff(lpx, rpx);
                        }
                    }
                    curCost[d] = sum;
                }

                /* find minPrev over prevCostL */
                cost_t minPrev = INF_COST;
            MinLoop:
                for (int d = 0; d < DISP; d++)
                {
                    if (prevCostL[d] < minPrev)
                        minPrev = prevCostL[d];
                }

                cost_t bestCost = INF_COST;
                disp_t bestDisp = 0;

            AggregationLoop:
                for (int d = 0; d < DISP; d++)
                {
                    cost_t p0 = prevCostL[d];
                    cost_t p1 = (d > 0) ? sat12(prevCostL[d-1] + P1) :
                    		INF_COST;
                    cost_t p2 = (d < DISP-1) ? sat12(prevCostL[d+1] + P1) :
                    		INF_COST;
                    cost_t p3 = sat12(minPrev + P2);

                    cost_t min_penalty = p0;
                    if (p1 < min_penalty) min_penalty = p1;
                    if (p2 < min_penalty) min_penalty = p2;
                    if (p3 < min_penalty) min_penalty = p3;

                    cost_t current_agg_cost = sat12(curCost[d] + min_penalty
                    		- minPrev);
                    aggCost[d] = current_agg_cost;

                    if (current_agg_cost < bestCost)
                    {
                        bestCost = current_agg_cost;
                        bestDisp = disp_t(d);
                    }
                }

                /* Copy aggregated costs to prevCostL for the next pixel */
            CopyPrev:
                for (int d = 0; d < DISP; ++d)
                {
                    prevCostL[d] = aggCost[d];
                }

                if (c >= DISP - 1)
                {
                    outDisp = bestDisp;
                }
            }

            disp.write(outDisp);
        }
    }
}
