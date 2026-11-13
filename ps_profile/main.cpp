#include <cmath>
#include "sgm_config.hpp"
#include "sgm_sw_core.hpp"
#include "dataset_frames.hpp"
#include "xil_printf.h"
#include "xtime_l.h"

#define CHECK_OUTPUT 0
#define PROFILE_STAGES 0


int main()
{
	static uint8_t disp_out[IMG_H][IMG_W];

	/* Evaluation metrics */
	int kitti_valid_frame = 0;
	int hw_valid_frame = 0;

	double total_kitti_d1_error = 0;
	double total_mae = 0;
	double total_Bad1_error = 0, total_Bad3_error = 0;
	int invalid_selected_count = 0;

	uint64_t total_latency = 0;

	uint64_t total_linebuffer = 0;
	uint64_t total_slidingwindow = 0;
	uint64_t total_census = 0;
	uint64_t total_aggregate = 0;
	uint64_t total_commit = 0;

	static double d1_per_frame[NUM_FRAMES];
	static double mae_per_frame[NUM_FRAMES];
	static double bad1_per_frame[NUM_FRAMES];
	static double bad3_per_frame[NUM_FRAMES];
	static uint32_t latency_per_frame[NUM_FRAMES];



	const int cx = WIN >> 1;
	const int cy = WIN >> 1;

	const int valid_r_min = WIN - 1 - cy;
	const int valid_r_max = IMG_H - 1 - cy;
	const int valid_c_max = IMG_W - cx;

	xil_printf("Starting Variant A - PS SGM evaluation... \r\n");

	sgm_sw_core(left_frames[0], right_frames[0], disp_out);

	XTime t0, t1;

	for(int i = 0; i < NUM_FRAMES; ++i)
	{
		if (i % 20 == 0)
			xil_printf("Frame %d\r\n", i);

		const float scale_x = scale_x_frames[i];

		XTime_GetTime(&t0);
		sgm_sw_core(left_frames[i], right_frames[i], disp_out);
		XTime_GetTime(&t1);

		total_linebuffer += t_linebuffer;
		total_slidingwindow += t_slidingwindow;
		total_census += t_computecensus;
		total_aggregate += t_aggregatecost;
		total_commit += t_commitcosts;

		uint64_t cycles = uint64_t(t1 - t0);
	    uint32_t latency_us = (uint32_t)((cycles * 1000000ULL) / COUNTS_PER_SECOND);
	    latency_per_frame[i] = latency_us;

	    if (i % 20 == 0)
	    	xil_printf("Latency = %u us\r\n", (unsigned int)latency_us);

	    total_latency += latency_us;
#if CHECK_OUTPUT
		volatile uint32_t checksum = 0;
		for(int r = 0; r < IMG_H; ++r)
		{
			for(int c = 0; c < IMG_W; ++c)
			{
				checksum +=disp_out[r][c];
			}
		}
	    xil_printf("checksum = %u \r\n", checksum);
#endif

		int kitti_d1_err = 0;
		int kitti_valid_count = 0;
		int kitti_invalid_count = 0;

		int hw_valid_count = 0, hw_invalid_count = 0;

		int bad1 = 0, bad3 = 0;
		double sum_abs_err = 0.0;

		for (int r = 0; r < IMG_H; ++r)
		{
			for (int c = 0; c < IMG_W; ++c)
			{
				float gt_disp = gt_frames[i][r][c];
				float est_disp = disp_out[r][c];

				if(r >= valid_r_min && est_disp > c)
					++invalid_selected_count;

				bool gt_valid = gt_disp > 0.0f;
				if(!gt_valid)
				{
					// Invalid kitti ground truth
					kitti_invalid_count++;
					continue;
				}
				++kitti_valid_count;
				float err = (std::abs(est_disp - gt_disp)) / scale_x;

				/* Kitt-style D1 */
				bool abs_error_gt3 = err > 3.0f;
				bool rel_error_gt5 = ((err / (gt_disp/scale_x)) > 0.05f);

				if(abs_error_gt3 && rel_error_gt5)
				{
					++kitti_d1_err;
				}

		        bool roi_valid = (r >= valid_r_min) && (r <= valid_r_max)
		        		&& (c < valid_c_max);

		        bool disp_range_valid = (gt_disp < DISP) && (gt_disp <= c);

		        bool hw_valid = roi_valid && disp_range_valid;

		    	if(!hw_valid)
		    	{
		    		++hw_invalid_count;
		    	}
		    	else
		    	{
			    	++hw_valid_count;
			    	sum_abs_err += err;
			    	if (err > 1.0f) bad1++;
			    	if (err > 3.0f) bad3++;
		    	}
		    }
		}

		double current_kitti_error = 0.0f;
		double current_mae = 0.0f;
		double current_bad1 = 0.0f;
		double current_bad3= 0.0f;

        if(kitti_valid_count > 0)
        {
        	current_kitti_error = (100.0 * kitti_d1_err / kitti_valid_count);
        	++kitti_valid_frame;
        	total_kitti_d1_error += current_kitti_error;
        }
        else
        {
            xil_printf( "ERROR: No KITTI-valid GT pixels at frame %d\n", i);
            return 5;
        }

        if(hw_valid_count > 0)
        {
        	++hw_valid_frame;
        	current_mae = (sum_abs_err / hw_valid_count);
			current_bad1 = (100.0 * bad1 / hw_valid_count);
			current_bad3= (100.0 * bad3 / hw_valid_count);

			total_mae += current_mae;
			total_Bad1_error += current_bad1;
			total_Bad3_error +=current_bad3;
        }
        if (hw_valid_count == 0)
        {
            xil_printf("ERROR: No HW-valid GT pixels for comparison at frame: %d\n", i);
            return 6;
        }

    	d1_per_frame[i] = current_kitti_error;
    	mae_per_frame[i] = current_mae;
    	bad1_per_frame[i] = current_bad1;
    	bad3_per_frame[i] = current_bad3;

        unsigned int d1_per_frame_x1000  = (unsigned int)((d1_per_frame[i] * 1000.0));
        unsigned int mae_per_frame_x1000  = (unsigned int)((mae_per_frame[i] * 1000.0));
        unsigned int bad1_per_frame_x1000  = (unsigned int)((bad1_per_frame[i] * 1000.0));
        unsigned int bad3_per_frame_x1000  = (unsigned int)((bad3_per_frame[i] * 1000.0));

        if (i % 20 == 0)
        {
        	xil_printf ("Frame %d, ", i);
        	xil_printf (" D1 = %u.%03u", d1_per_frame_x1000/1000, d1_per_frame_x1000%1000);
        	xil_printf (" MAE = %u.%03u", mae_per_frame_x1000/1000, mae_per_frame_x1000%1000);
        	xil_printf (" Bad(>1) = %u.%03u", bad1_per_frame_x1000/1000, bad1_per_frame_x1000%1000);
        	xil_printf (" Bad(>3) = %u.%03u\n", bad3_per_frame_x1000/1000, bad3_per_frame_x1000%1000);
        }

        if((kitti_invalid_count + kitti_valid_count) != (IMG_H * IMG_W))
        {
        	xil_printf("ERROR: GT validity counts do not match image "
        			"size at frame: %d\n", i);
        }

        if((hw_invalid_count + hw_valid_count) != kitti_valid_count)
        {
        	xil_printf ("ERROR: HW-valid + HW-excluded counts do not match"
        			" KITTI-valid count at frame: %d", i);
        }
	}

	if(NUM_FRAMES > 0)
	{
		xil_printf("---Mean Latency--- \n ");

		double mean_latency = static_cast<double>(total_latency) / NUM_FRAMES;
		double squared_diff_sum = 0.0;

		for(int i = 0; i < NUM_FRAMES; ++i)
		{
			double diff = latency_per_frame[i] - mean_latency;
			squared_diff_sum += diff * diff;
		}
		double std_latency = std::sqrt(squared_diff_sum / NUM_FRAMES);

		uint32_t mean_latency_x1000  = (uint32_t)((mean_latency * 1000.0));
		uint32_t std_latency_x1000  = (uint32_t)((std_latency * 1000.0));
        uint64_t fps_x1000 = ((uint64_t)NUM_FRAMES * 1000000000ULL) / total_latency;


    	xil_printf("Mean latency = %u.%03u us\r\n",
    			mean_latency_x1000/1000, mean_latency_x1000%1000);

    	xil_printf("Latency Standard Deviation = %u.%03u us\r\n",
    			std_latency_x1000/1000, std_latency_x1000%1000);

    	xil_printf("Throughput = %u.%03u FPS\r\n",
    			(unsigned int)(fps_x1000 / 1000), (unsigned int)(fps_x1000 % 1000));

#if PROFILE_STAGES
    	uint64_t total_profile_ticks = total_linebuffer + total_slidingwindow +
    			total_census + total_aggregate + total_commit;

    	double avg_linebuffer_ms = ((double) total_linebuffer / NUM_FRAMES)
    			* 1000.0 / COUNTS_PER_SECOND;

    	double avg_slidingwindow_ms = ((double) total_slidingwindow / NUM_FRAMES)
    			* 1000.0 / COUNTS_PER_SECOND;

    	double avg_census_ms = ((double) total_census / NUM_FRAMES)
    			* 1000.0 / COUNTS_PER_SECOND;

    	double avg_aggregate_ms = ((double) total_aggregate / NUM_FRAMES)
    			* 1000.0 / COUNTS_PER_SECOND;

    	double avg_commit_ms = ((double) total_commit / NUM_FRAMES)
    			* 1000.0 / COUNTS_PER_SECOND;


    	double pct_linebuffer = (100.0 * total_linebuffer) /
    			total_profile_ticks;

    	double pct_slidingwindow = (100.0 * total_slidingwindow) /
    			total_profile_ticks;

    	double pct_census = (100.0 * total_census) /
    			total_profile_ticks;

    	double pct_aggregate = (100.0 * total_aggregate) /
    			total_profile_ticks;

    	double pct_commit = (100.0 * total_commit) /
    			total_profile_ticks;

    	uint32_t pct_linebuffer_x100	= (uint32_t)(100.0 * pct_linebuffer);
    	uint32_t pct_slidingwidnow_x100 = (uint32_t)(100.0 * pct_slidingwindow);
    	uint32_t pct_census_x100		= (uint32_t)(100.0 * pct_census);
    	uint32_t pct_aggregate_x100		= (uint32_t)(100.0 * pct_aggregate);
    	uint32_t pct_commit_x100		= (uint32_t)(100.0 * pct_commit);


    	uint32_t linebuffer_x1000    	= (uint32_t)(1000.0 * avg_linebuffer_ms);
    	uint32_t slidingwindow_ms_x1000 = (uint32_t)(1000.0 * avg_slidingwindow_ms);
    	uint32_t census_ms_x1000		= (uint32_t)(1000.0 * avg_census_ms);
    	uint32_t aggregate_ms_x1000 	= (uint32_t)(1000.0 * avg_aggregate_ms);
    	uint32_t commit_ms_x1000		= (uint32_t)(1000.0 * avg_commit_ms);



    	xil_printf("Line buffer:\r\n");
    	xil_printf("Mean = %u.%03u ms/frame\r\n",
    	    	linebuffer_x1000/1000, linebuffer_x1000%1000);
    	xil_printf("Share = %u.%02u %%\r\n",
    	    	pct_linebuffer_x100/100, pct_linebuffer_x100%100);

    	xil_printf("\r\n\r\n Sliding window:\r\n");
    	xil_printf("Mean = %u.%03u ms/frame\r\n",
    			slidingwindow_ms_x1000/1000, slidingwindow_ms_x1000%1000);
    	xil_printf("Share = %u.%02u %%\r\n",
    			pct_slidingwidnow_x100/100, pct_slidingwidnow_x100%100);

    	xil_printf("\r\n\r\n Census cost:\r\n");
    	xil_printf("Mean = %u.%03u ms/frame\r\n",
    			census_ms_x1000/1000, census_ms_x1000%1000);
    	xil_printf("Share = %u.%02u %%\r\n",
    			pct_census_x100/100, pct_census_x100%100);

    	xil_printf("\r\n\r\n SGM aggregation:\r\n");
    	xil_printf("Mean = %u.%03u ms/frame\r\n",
    			aggregate_ms_x1000/1000, aggregate_ms_x1000%1000);
    	xil_printf("Share = %u.%02u %%\r\n",
    			pct_aggregate_x100/100, pct_aggregate_x100%100);

    	xil_printf("\r\n\r\n Commit:\r\n");
    	xil_printf("Mean = %u.%03u ms/frame\r\n",
    			commit_ms_x1000/1000, commit_ms_x1000%1000);
    	xil_printf("Share = %u.%02u %%\r\n",
    			pct_commit_x100/100, pct_commit_x100%100);
#endif
	}

	if(kitti_valid_frame > 0)
	{
		xil_printf("---KITTI-style Evaluation --- \n ");

		double mean_d1 = total_kitti_d1_error / kitti_valid_frame;
		double squared_diff_sum = 0.0;

		for(int i = 0; i < NUM_FRAMES; ++i)
		{
			double diff = d1_per_frame[i] - mean_d1;
			squared_diff_sum += diff * diff;
		}
		double std_d1 = std::sqrt(squared_diff_sum / kitti_valid_frame);

        unsigned int mean_d1_x1000  = (unsigned int)((mean_d1 * 1000.0));
        unsigned int std_d1_x1000  = (unsigned int)((std_d1 * 1000.0));

    	xil_printf("Mean D1 = %u.%03u\r\n", mean_d1_x1000/1000, mean_d1_x1000%1000);
    	xil_printf("D1 Standard Deviation = %u.%03u\r\n", std_d1_x1000/1000, std_d1_x1000%1000);
	}

	if(hw_valid_frame > 0)
	{
		xil_printf("\n --- HW Valid Evaluation --- \n ");

		double mean_mae = total_mae / hw_valid_frame;
		double mae_squared_diff_sum = 0.0;

		double mean_bad1 = total_Bad1_error / hw_valid_frame;
		double bad1_squared_diff_sum = 0.0;

		double mean_bad3 = total_Bad3_error / hw_valid_frame;
		double bad3_squared_diff_sum = 0.0;

		for(int i = 0; i < NUM_FRAMES; ++i)
		{
			double mae_diff = mae_per_frame[i] - mean_mae;
			mae_squared_diff_sum += mae_diff * mae_diff;


			double bad1_diff = bad1_per_frame[i] - mean_bad1;
			bad1_squared_diff_sum += bad1_diff * bad1_diff;


			double bad3_diff = bad3_per_frame[i] - mean_bad3;
			bad3_squared_diff_sum += bad3_diff * bad3_diff;
		}
		double std_mae = std::sqrt(mae_squared_diff_sum / hw_valid_frame);
		double std_bad1 = std::sqrt(bad1_squared_diff_sum / hw_valid_frame);
		double std_bad3 = std::sqrt(bad3_squared_diff_sum / hw_valid_frame);

        unsigned int mean_mae_x1000  = (unsigned int)((mean_mae * 1000.0));
        unsigned int std_mae_x1000  = (unsigned int)((std_mae * 1000.0));

        unsigned int mean_bad1_x1000  = (unsigned int)((mean_bad1 * 1000.0));
        unsigned int std_bad1_x1000  = (unsigned int)((std_bad1 * 1000.0));

        unsigned int mean_bad3_x1000  = (unsigned int)((mean_bad3 * 1000.0));
        unsigned int std_bad3_x1000  = (unsigned int)((std_bad3 * 1000.0));

    	xil_printf("Mean MAE = %u.%03u\r\n", mean_mae_x1000/1000, mean_mae_x1000%1000);
    	xil_printf("MAE Standard Deviation = %u.%03u\r\n", std_mae_x1000/1000, std_mae_x1000%1000);

    	xil_printf("Mean Bad(>1) = %u.%03u\r\n", mean_bad1_x1000/1000, mean_bad1_x1000%1000);
    	xil_printf("Bad(>1) Standard Deviation = %u.%03u\r\n", std_bad1_x1000/1000, std_bad1_x1000%1000);

    	xil_printf("Mean Bad(>3) = %u.%03u\r\n", mean_bad3_x1000/1000, mean_bad3_x1000%1000);
    	xil_printf("Bad(>3) Standard Deviation = %u.%03u\r\n", std_bad3_x1000/1000, std_bad3_x1000%1000);
	}









    while(1);
}
