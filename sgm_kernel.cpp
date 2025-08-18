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

    /* Line buffer */
    xf::cv::LineBuffer<WIN, IMG_W, pix_t> bufL;
    xf::cv::LineBuffer<WIN, IMG_W, pix_t> bufR;

    /* WIN×WIN windows (registers) */
    xf::cv::Window<WIN, WIN, pix_t> winL;
    xf::cv::Window<WIN, WIN, pix_t> winR;

    /* Per-row rolling costs (left→right) */
    static cost_t prevCostL[DISP];
#pragma HLS bind_storage variable=prevCostL type=RAM_1P impl=LUTRAM
#pragma HLS ARRAY_PARTITION variable=prevCostL complete dim=1

    static cost_t curCost[DISP];
#pragma HLS bind_storage variable=curCost type=RAM_1P impl=LUTRAM
#pragma HLS ARRAY_PARTITION variable=curCost complete dim=1

    /* For a WIN×WIN window, center offsets */
    const int cy = WIN >> 1; // 1 when WIN=3
    const int cx = WIN >> 1; // 1 when WIN=3

Row:
    for (int r = 0; r < IMG_H; ++r) {
        /* Reset aggregation for new row */
        for (int d = 0; d < DISP; ++d) {
        #pragma HLS UNROLL
            prevCostL[d] = cost_t(4095);
        }

    Col:
        for (int c = 0; c < IMG_W; ++c) {
        #pragma HLS PIPELINE II=1

            /* 1) Stream next pixels */
            pix_t pL = left.read();
            pix_t pR = right.read();

            /* 2) Update line buffers */
            bufL.shift_up(c);
            bufR.shift_up(c);
            bufL.insert_bottom(pL, c);
            bufR.insert_bottom(pR, c);

            /* Default output (warm‑up) */
            pix_t outDisp = 0;

            /* 3) Windows valid after WIN-1 rows */
            if (r >= WIN - 1) {
                /* Slide windows and insert new rightmost column */
                winL.shift_pixels_left();
                winR.shift_pixels_left();
                for (int k = 0; k < WIN; ++k) {
                #pragma HLS UNROLL
                    winL.insert_pixel(bufL.getval(k, c), k, WIN - 1);
                    winR.insert_pixel(bufR.getval(k, c), k, WIN - 1);
                }

                /* Accumulate min of previous-column costs */
                cost_t minPrev = cost_t(4095);

            DispLoop:
                for (int d = 0; d < DISP; ++d) {
                #pragma HLS UNROLL
                    /* ---------- 3×3 SAD around (r,c) vs (r,c-d) ---------- */
                    cost_t sum = 0;
                    for (int wy = 0; wy < WIN; ++wy) {
                    #pragma HLS UNROLL
                        for (int wx = 0; wx < WIN; ++wx) {
                        #pragma HLS UNROLL
                            /* Left window pixel (already in registers) */
                            pix_t lpx = winL.getval(wy, wx);

                            /* Column on right image for this tap */
                            int col_r = (c - d) + (wx - cx);

                            /* Fetch from the right line buffer (row wy, col col_r) */
                            pix_t rpx = (col_r >= 0 && col_r < IMG_W)
                                            ? bufR.getval(wy, col_r)
                                            : pix_t(0);

                            sum += absdiff(lpx, rpx);
                        }
                    }
                    curCost[d] = sum; /* fits 12 bits: 9*255 = 2295 */

                    /* ---------- SGM L→R neighbors (prev column) ---------- */
                    cost_t p0 = prevCostL[d];
                    cost_t p1 = d            ? sat12(prevCostL[d-1] + P1) : cost_t(4095);
                    cost_t p2 = (d < DISP-1) ? sat12(prevCostL[d+1] + P1) : cost_t(4095);

                    if (p0 < minPrev) minPrev = p0;
                    if (p1 < minPrev) minPrev = p1;
                    if (p2 < minPrev) minPrev = p2;
                }

                /* Update aggregated costs & choose best disparity */
                cost_t bestCost = cost_t(4095);
                pix_t  bestDisp = 0;

            UpdateLoop:
                for (int d = 0; d < DISP; ++d) {
                #pragma HLS UNROLL
                    cost_t thresh   = sat12(minPrev + P2);
                    cost_t prevBest = (prevCostL[d] < thresh) ? prevCostL[d] : thresh;
                    cost_t agg      = sat12(curCost[d] + prevBest);

                    prevCostL[d] = agg;

                    if (agg < bestCost) {
                        bestCost = agg;
                        bestDisp = pix_t(d);
                    }
                }

                /* Promote to valid disparity once both margins are met */
                if (c >= (WIN - 1) && c >= (DISP - 1)) {
                    outDisp = bestDisp;
                }
            }

            /* 4) Always write one output per input */
            disp.write(outDisp);
        }
    }
}
