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
#pragma HLS DATAFLOW

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

    static cost_t prevCostT[IMG_W][DISP];
#pragma HLS bind_storage variable=prevCostT type=RAM_1P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=prevCostT complete dim=2

    static cost_t curCost[DISP];
#pragma HLS bind_storage variable=curCost type=RAM_1P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=curCost complete dim=1

    static cost_t aggCost[DISP];
#pragma HLS bind_storage variable=aggCost type=RAM_1P impl=BRAM
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
		#pragma HLS UNROLL
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
			#pragma HLS UNROLL
				prevCostT[c][d] = cost_t(0);
			}

			}

        }

    Col:
        for (int c = 0; c < IMG_W; c++)
        {
		#pragma HLS PIPELINE II=1
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
				#pragma HLS UNROLL
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
                cost_t minPrevLR = INF_COST;
            MinLoopLR:
                for (int d = 0; d < DISP; d++)
                {
				#pragma HLS UNROLL
                    if (prevCostL[d] < minPrevLR)
                    	minPrevLR = prevCostL[d];
                }

                cost_t minPrevTB = INF_COST;
            MinLoopTB:
                  for (int d = 0; d < DISP; d++)
                  {
				#pragma HLS UNROLL
                      cost_t v = prevCostT[c][d];
                      if (v < minPrevTB)
                          minPrevTB = v;
                  }

                cost_t bestCost = INF_COST;
                disp_t bestDisp = 0;

            AggregationLoop:
                for (int d = 0; d < DISP; d++)
                {
				#pragma HLS UNROLL
                    cost_t p0_LR = prevCostL[d];
                    cost_t p1_LR = (d > 0) ? sat12(prevCostL[d-1] + P1) :
                    		INF_COST;
                    cost_t p2_LR = (d < DISP-1) ? sat12(prevCostL[d+1] + P1) :
                    		INF_COST;
                    cost_t p3_LR = sat12(minPrevLR + P2);

                    cost_t minLR = p0_LR;
                    if (p1_LR < minLR) minLR = p1_LR;
                    if (p2_LR < minLR) minLR = p2_LR;
                    if (p3_LR < minLR) minLR = p3_LR;

                    cost_t aggLR = sat12(curCost[d] + minLR
                    		- minPrevLR);
                    aggLR_arr[d] = aggLR;


                    cost_t p0_TB = prevCostT[c][d];
                    cost_t p1_TB = (d > 0)      ? sat12(prevCostT[c][d-1] + P1) :
                    		INF_COST;
                    cost_t p2_TB = (d < DISP-1) ? sat12(prevCostT[c][d+1] + P1) :
                    		INF_COST;
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

                /* Copy aggregated costs to prevCostL for the next pixel */
            CopyPrevLR:
                for (int d = 0; d < DISP; ++d)
                {
				#pragma HLS UNROLL
                    prevCostL[d] = aggLR_arr[d];
                    prevCostT[c][d] = aggTB_arr[d];
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
