#include "sgm_params.hpp"

/* Penalties */
const cost_t P1 = cost_t(10);
const cost_t P2 = cost_t(150);

void sgm_kernel(hls::stream<pix_t>& left,
                hls::stream<pix_t>& right,
                hls::stream<pix_t>& disp)
{
#pragma HLS INTERFACE axis         port=left   register
#pragma HLS INTERFACE axis         port=right  register
#pragma HLS INTERFACE axis         port=disp   register
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS DATAFLOW

    /* Line buffers for the left & right images */
    xf::cv::LineBuffer<CENSUS_WIN, IMG_W, pix_t> bufL;
    xf::cv::LineBuffer<CENSUS_WIN, IMG_W, pix_t> bufR;

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

    /* Reverse-pass state (R->L, B->T) */

    static cost_t prevCostR[DISP];
#pragma HLS bind_storage variable=prevCostR type=RAM_1P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=prevCostR complete dim=1

    static cost_t prevCostB[IMG_W][DISP];
#pragma HLS bind_storage variable=prevCostB type=RAM_1P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=prevCostB complete dim=2

    static cost_t partialCost[IMG_H][IMG_W][DISP];
#pragma HLS bind_storage variable=partialCost type=RAM_2P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=partialCost complete dim=3


    static pix_t imgL[IMG_H][IMG_W];
    static pix_t imgR[IMG_H][IMG_W];
#pragma HLS bind_storage variable=imgL type=RAM_2P impl=BRAM
#pragma HLS bind_storage variable=imgR type=RAM_2P impl=BRAM

    static pix_t dispBuf[IMG_H][IMG_W];
#pragma HLS bind_storage variable=dispBuf type=RAM_2P impl=BRAM




    /* center offsets */
    const int cy = CENSUS_CY;
    const int cx = CENSUS_CX;

RowFwd:
    for (int r = 0; r < IMG_H; r++)
    {
        /* Reset aggregation for new row */
    ResetCostsFwd:
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

    ColFwd:
        for (int c = 0; c < IMG_W; c++)
        {
		#pragma HLS PIPELINE II=1
		#pragma HLS DEPENDENCE variable=bufL inter false
		#pragma HLS DEPENDENCE variable=bufR inter false

            /* Stream next pixels */
            pix_t pL = left.read();
            pix_t pR = right.read();

            imgL[r][c] = pL;
            imgR[r][c] = pR;

            /* Update left line buffer at column c */
            bufL.shift_up(c);
            bufL.insert_bottom(pL, c);
            bufR.shift_up(c);
            bufR.insert_bottom(pR, c);

            /* Default output */
            pix_t outDisp = 0;

            if (r >= WIN - 1)
            {
                /* Calculate census matching cost for all disparities */
            Census_LoopFwd:
                for (int d = 0; d < DISP; d++)
                {

                	pix_t centerL = bufL.getval(cy, c);
                	pix_t centerR = pix_t(0);
				#pragma HLS UNROLL
                    cost_t sum = 0;
                WinYFwd:
                    for (int wy = 0; wy < CENSUS_WIN; wy++)
                    {
					#pragma HLS UNROLL
                    WinXFwd:
                        for (int wx = 0; wx < CENSUS_WIN; wx++)
                        {
                        #pragma HLS UNROLL
                            int colL = c + wx - cx;
                            int colR = (c - d) + wx - cx;

                            pix_t lpx = pix_t(0);
                            pix_t rpx = pix_t(0);

                            if (colL >= 0 && colL < IMG_W)
                                lpx = bufL.getval(wy, colL);

                            if (colR >= 0 && colR < IMG_W)
                                rpx = bufR.getval(wy, colR);

                            bool bitL = (lpx < centerL);
                            bool bitR = (rpx < centerR);

                            sum += (bitL ^ bitR);
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

            AggregationLoopFwd:
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
                    cost_t p1_TB = (d > 0) ? sat12(prevCostT[c][d-1] + P1) :
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
                }

                /* Copy aggregated costs to prevCostL for the next pixel */
            CopyPrevLRFwd:
                for (int d = 0; d < DISP; ++d)
                {
				#pragma HLS UNROLL
                    prevCostL[d] = aggLR_arr[d];
                    prevCostT[c][d] = aggTB_arr[d];
                    partialCost[r][c][d] = aggCost[d];
                }

               }
            }

        }

        InitBT:
            for (int c = 0; c < IMG_W; ++c)
            {
        	#pragma HLS LOOP_TRIPCOUNT min=IMG_W max=IMG_W
              for (int d = 0; d < DISP; ++d)
              {
			  #pragma HLS UNROLL
                prevCostB[c][d] = cost_t(0);
              }
            }

         RowRev:
             for (int r = IMG_H - 1; r >= 0; --r)
                {
                ResetCostsRev:
                    for (int d = 0; d < DISP; d++)
                    {
            		#pragma HLS UNROLL
                        prevCostR[d] = cost_t(0);
                    }

                ColRev:
                    for (int c = IMG_W - 1; c >= 0; --c)
                    {
                    #pragma HLS PIPELINE II=1
                    	pix_t outDisp = 0;

                    	if (r >= WIN - 1)
                    	{

                    Census_LoopRev:
						for (int d = 0; d < DISP; d++)
						{
                    	#pragma HLS UNROLL
							pix_t centerL = imgL[r][c];
							pix_t centerR = pix_t(0);

							int centerR_col = c - d;
							if (centerR_col >= 0 && centerR_col << IMG_W)
								centerR = imgR[r][centerR_col];
							cost_t sum = 0;

							for(int wy = 0; wy < CENSUS_WIN; wy++)
							{
							#pragma HLS UNROLL
								for(int wx = 0; wx < CENSUS_WIN; wx++)
								{
								#pragma HLS UNROLL
									int rr = r - (CENSUS_WIN - 1) + wy;
									int ccL = c + wx - cx;
									int ccR = (c - d) + wx - cx;

							        pix_t lpx = pix_t(0);
							        pix_t rpx = pix_t(0);

							        if (rr >= 0 && rr < IMG_H && ccL >= 0 && ccL < IMG_W)
							                    lpx = imgL[rr][ccL];

							        if (rr >= 0 && rr < IMG_H && ccR >= 0 && ccR < IMG_W)
							                    rpx = imgR[rr][ccR];

							        bool bitL = (lpx < centerL);
							        bool bitR = (rpx < centerR);

							        sum += (bitL ^ bitR);
						         }
						     }
						    curCost[d] = sum;
						}
		                cost_t minPrevRL = min_array(prevCostR, DISP);
		                cost_t minPrevBT = min_array(prevCostB[c], DISP);

		                cost_t bestCost = INF_COST;
		                disp_t bestDisp = 0;

		            AggregationLoopRev:
					for (int d = 0; d < DISP; d++)
					{
					#pragma HLS UNROLL
						cost_t p0_RL = prevCostR[d];
						cost_t p1_RL = (d > 0) ? sat12(prevCostR[d-1] + P1) : INF_COST;
						cost_t p2_RL = (d < DISP-1) ? sat12(prevCostR[d+1] + P1) : INF_COST;
						cost_t p3_RL = sat12(minPrevRL + P2);
						cost_t minRL = min4(p0_RL, p1_RL, p2_RL, p3_RL);
						cost_t aggRL = sat12(curCost[d] + minRL - minPrevRL);

	                    cost_t p0_BT = prevCostB[c][d];
	                    cost_t p1_BT = (d > 0)      ? sat12(prevCostB[c][d-1] + P1) : INF_COST;
	                    cost_t p2_BT = (d < DISP-1) ? sat12(prevCostB[c][d+1] + P1) : INF_COST;
	                    cost_t p3_BT = sat12(minPrevBT + P2);
	                    cost_t minBT = min4(p0_BT, p1_BT, p2_BT, p3_BT);
	                    cost_t aggBT = sat12(curCost[d] + minBT - minPrevBT);


	                    cost_t fwd = partialCost[r][c][d];
	                    cost_t sum4 = sat12( sat12(fwd + aggRL) + aggBT );

	                    if (sum4 < bestCost)
	                    {
	                    	bestCost = sum4; bestDisp = disp_t(d);
	                    }
                    	prevCostR[d]     = aggRL;
                    	prevCostB[c][d]  = aggBT;
                   }
                   if (c >= DISP - 1)
                   {
                	   outDisp = bestDisp;
                   }
               }

               dispBuf[r][c] = outDisp;
             }
      }
   FlushDisp:
   for (int rr = 0; rr < IMG_H; ++rr)
   {
	   for (int cc = 0; cc < IMG_W; ++cc)
	   {
		   #pragma HLS PIPELINE II=1
		   	   disp.write(dispBuf[rr][cc]);
        }
   }
}
