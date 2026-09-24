#include "sgm_kernel-tb.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <sstream>
#include <iomanip>

int main()
{
	std::string left_dir = "/home/hamzy/SGM/Kitti-Data/training/image_2";
	std::string right_dir = "/home/hamzy/SGM/Kitti-Data/training/image_3";
	std::string disp_dir = "/home/hamzy/SGM/Kitti-Data/training/disp_noc_0";

	int kitti_valid_frame = 0;
	int hw_valid_frame = 0;

	double total_kitti_d1_error = 0;
	double total_mae = 0;
	double total_Bad1_error = 0, total_Bad3_error = 0;

	double d1_per_frame[200];
	double mae_per_frame[200];
	double bad1_per_frame[200];
	double bad3_per_frame[200];


	for(int i = 0; i < 200; ++i)
	{
		std::ostringstream ss;
		ss << std::setw(6) << std::setfill('0') << i;
		std::string id = ss.str();
		std::string file_name = id + "_10.png";

		std::string left_path  =  left_dir + "/" + file_name;
		std::string right_path = right_dir + "/" + file_name;
		std::string gt_path = disp_dir + "/" + file_name;

		/* Load as grayscale */
		cv::Mat left  = cv::imread(left_path,  cv::IMREAD_GRAYSCALE);
		cv::Mat right = cv::imread(right_path, cv::IMREAD_GRAYSCALE);
		cv::Mat gt = cv::imread(gt_path, cv::IMREAD_UNCHANGED);
		if (left.empty() || right.empty() || gt.empty())
		{
			std::cerr << "ERROR: Could not load input images:\n  "
					  << left_path << "\n  " << right_path <<  "\n  " <<
					  gt_path << std::endl;
			return 2;
		}

		/* Ensure size matches kernel shape */
		if (left.cols != IMG_W || left.rows != IMG_H)
		{
			cv::resize(left,  left,  cv::Size(IMG_W, IMG_H), 0, 0, cv::INTER_AREA);
		}
		if (right.cols != IMG_W || right.rows != IMG_H)
		{
			cv::resize(right, right, cv::Size(IMG_W, IMG_H), 0, 0, cv::INTER_AREA);
		}

		const float scale_x = static_cast<float>(IMG_W) /
				static_cast<float>(gt.cols);

		cv::Mat gt_f;
		gt.convertTo(gt_f, CV_32F, 1.0f / 256.0f);

		if(gt_f.cols != IMG_W || gt_f.rows != IMG_H)
		{
			cv::resize(gt_f, gt_f, cv::Size(IMG_W, IMG_H), 0, 0, cv::INTER_NEAREST);
			gt_f *= scale_x;
		}

		if (left.cols != IMG_W || left.rows != IMG_H ||
			right.cols != IMG_W || right.rows != IMG_H ||
			gt_f.cols != IMG_W || gt_f.rows != IMG_H)
		{
			std::cerr << "ERROR: Size mismatch after resize. "
					  << "Expected (" << IMG_W << "x" << IMG_H << ").\n";
			return 3;
		}

		/* Prepare memory-mapped input/output arrays */
		static bram_word_t left_arr[FRAME_WORDS];
		static bram_word_t right_arr[FRAME_WORDS];
		static bram_word_t disp_arr[FRAME_WORDS];

		for (int j = 0; j < FRAME_WORDS; ++j)
		{
			left_arr[j] = 0;
			right_arr[j] = 0;
			disp_arr[j] = 0;
		}

		for (int r = 0; r < IMG_H; ++r)
		{
			const uint8_t* lp = left.ptr<uint8_t>(r);
			const uint8_t* rp = right.ptr<uint8_t>(r);

			for (int c = 0; c < IMG_W; ++c)
			{
				int pixel_idx = r * IMG_W + c;
				int word_idx  = pixel_idx >> 2;
				int byte_idx  = pixel_idx & 3;

				left_arr[word_idx].range(byte_idx * 8 + 7,
										 byte_idx * 8) = lp[c];

				right_arr[word_idx].range(byte_idx * 8 + 7,
										  byte_idx * 8) = rp[c];
			}
		}

		/* Run kernel */
		sgm_kernel(left_arr, right_arr, disp_arr);

		/* Retrieve output disparity */
		cv::Mat disp(IMG_H, IMG_W, CV_8U);
		using out_u_t = uint8_t;

		for (int r = 0; r < IMG_H; ++r)
		{

		uint8_t *dp = disp.ptr<uint8_t>(r);

			for (int c = 0; c < IMG_W; ++c)
			{
				int pixel_idx = r * IMG_W + c;
				int word_idx  = pixel_idx >> 2;
				int byte_idx  = pixel_idx & 3;

				bram_word_t word = disp_arr[word_idx];

				dp[c] = static_cast<out_u_t>(
					word.range(byte_idx * 8 + 7,
							   byte_idx * 8));
			}
		}
		/* Evaluation metrics */
			int kitti_d1_err = 0;
			int kitti_valid_count = 0;
			int kitti_invalid_count = 0;

			int hw_valid_count = 0, hw_invalid_count = 0;

			int bad1 = 0, bad3 = 0;
			double sum_abs_err = 0.0;

			const int cx = WIN >> 1;

			const int valid_r_min = WIN - 1;
			const int valid_c_min = (DISP - 1) + cx;
			const int valid_c_max = IMG_W - cx;

			for (int r = 0; r < IMG_H; ++r)
			{
				for (int c = 0; c < IMG_W; ++c)
				{
					float est_disp = float(disp.at<out_u_t>(r,c));
					float gt_disp = gt_f.at<float>(r,c);

					bool gt_valid = gt_disp > 0.0f;
					if(!gt_valid)
					{
						// Invalid kitti ground truth
						kitti_invalid_count++;
						continue;
					}
					++kitti_valid_count;
					float err = std::abs(est_disp - gt_disp);

					/* Kitt-style D1 */
					bool abs_error_gt3 = err > 3.0f;
					bool rel_error_gt5 = ((err / gt_disp) > 0.05f);

					if(abs_error_gt3 && rel_error_gt5)
					{
						++kitti_d1_err;
					}

					bool roi_valid =
						(r >= valid_r_min) &&
						(c >= valid_c_min) &&
						(c < valid_c_max);

					bool disp_range_valid = (gt_disp < DISP);

					bool hw_valid = roi_valid && disp_range_valid;

					if(!hw_valid)
						{
							++hw_invalid_count;
							continue;
						}
					++hw_valid_count;

					sum_abs_err += err;
					if (err > 1.0f) bad1++;
					if (err > 3.0f) bad3++;
				}
			}
			////////////////////////////////
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
	            std::cerr << "ERROR: No KITTI-valid GT pixels at frame "
	                      << id << std::endl;
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
	            std::cerr << "ERROR: No HW-valid GT pixels for comparison at frame: "
	            		<< id << std::endl;;
	            return 6;
	        }

	    	d1_per_frame[i] = current_kitti_error;
	    	mae_per_frame[i] = current_mae;
	    	bad1_per_frame[i] = current_bad1;
	    	bad3_per_frame[i] = current_bad3;

			std::cout << "Frame " << id << ", D1 = " << d1_per_frame[i] << " %"
					", MAE = " << mae_per_frame[i] << " px" <<", Bad1 = " <<
					bad1_per_frame[i] << " %" << ", Bad3 = " <<
					bad3_per_frame[i] << " %" <<std::endl;

	        if((kitti_invalid_count + kitti_valid_count) != (IMG_H * IMG_W))
	        {
	        	std::cerr << "ERROR: GT validity counts do not match image "
	        			"size at frame: "
	        			<< id << std::endl;
	        }

	        if((hw_invalid_count + hw_valid_count) != kitti_valid_count)
	        {
	        	std::cerr << "ERROR: HW-valid + HW-excluded counts do not match"
	        			" KITTI-valid count at frame: " << id << std::endl;
	        }
	}
////////////////////////////////////////////////////////////////////////
	double max_d1 = d1_per_frame[0], max_mae = mae_per_frame[0];
	double min_d1 = d1_per_frame[0], min_mae = mae_per_frame[0];

	double max_bad1 = bad1_per_frame[0], max_bad3 = bad3_per_frame[0];
	double min_bad1 = bad1_per_frame[0], min_bad3 = bad3_per_frame[0];

	int max_d1_id = 0 , max_mae_id = 0, max_bad1_id = 0, max_bad3_id = 0;
	int min_d1_id = 0, min_mae_id = 0, min_bad1_id = 0, min_bad3_id = 0;
	/////////////////////////////////////////////////////////////////////////////
	for(int i  = 0; i < 200; ++i)
	{
		if(min_d1 > d1_per_frame[i])
		{
			min_d1 = d1_per_frame[i];
			min_d1_id = i;
		}

		if(max_d1 < d1_per_frame[i])
		{
			max_d1 = d1_per_frame[i];
			max_d1_id = i;
		}

		if(min_mae > mae_per_frame[i])
		{
			min_mae = mae_per_frame[i];
			min_mae_id = i;
		}

		if(max_mae < mae_per_frame[i])
		{
			max_mae = mae_per_frame[i];
			max_mae_id = i;
		}

		if(min_bad1 > bad1_per_frame[i])
		{
			min_bad1 = bad1_per_frame[i];
			min_bad1_id = i;
		}

		if(max_bad1 < bad1_per_frame[i])
		{
			max_bad1 = bad1_per_frame[i];
			max_bad1_id = i;
		}

		if(min_bad3 > bad3_per_frame[i])
		{
			min_bad3 = bad3_per_frame[i];
			min_bad3_id = i;
		}

		if(max_bad3 < bad3_per_frame[i])
		{
			max_bad3 = bad3_per_frame[i];
			max_bad3_id = i;
		}
	}

	if(kitti_valid_frame > 0)
	{
		std::cout << "\n ---KITTI-style Evaluation --- \n ";

		double mean_d1 = total_kitti_d1_error / kitti_valid_frame;
		double squared_diff_sum = 0.0;

		for(int i = 0; i < 200; ++i)
		{
			double diff = d1_per_frame[i] - mean_d1;
			squared_diff_sum += diff * diff;

		}

		double std_d1 = std::sqrt(squared_diff_sum / kitti_valid_frame);

		std::cout << "Mean D1: " << mean_d1 << " %\n";

		std::cout << "D1 Standard Deviation: " << std_d1 << " %" <<std::endl;

		std::cout << "Minimum D1: " << min_d1 << " % (Frame: "
				<< min_d1_id << ")" <<std::endl;

		std::cout << "Maximum D1: " << max_d1 << " % (Frame: "
				<< max_d1_id << ")" << std::endl;
	}

	if(hw_valid_frame > 0)
	{
		std::cout << "\n --- HW Valid Evaluation --- \n ";

		double mean_mae = total_mae / hw_valid_frame;
		double mae_squared_diff_sum = 0.0;

		double mean_bad1 = total_Bad1_error / hw_valid_frame;
		double bad1_squared_diff_sum = 0.0;

		double mean_bad3 = total_Bad3_error / hw_valid_frame;
		double bad3_squared_diff_sum = 0.0;

		for(int i = 0; i < 200; ++i)
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


    	std::cout << "Mean MAE: " << mean_mae << " px\n";
		std::cout << "MAE Standard Deviation: " << std_mae << " px" << std::endl;

    	std::cout << "Mean Bad >1 px: " << mean_bad1 << " %\n";
    	std::cout << "Bad > 1 Standard Deviation: " << std_bad1 << " %" << std::endl;

    	std::cout << "Mean Bad >3 px: " << mean_bad3 << " %\n";
    	std::cout << "Bad > 3 Standard Deviation: " << std_bad3 << " %" << std::endl;


		std::cout << "Minimum MAE: " << min_mae << " px (Frame: "
				<< min_mae_id << ")" << std::endl;

		std::cout << "Maximum MAE: " << max_mae << " px (Frame: "
				<< max_mae_id << ")" <<std::endl;


		std::cout << "Minimum Bad > 1: " << min_bad1 << " % (Frame: "
				<< min_bad1_id << ")" <<std::endl;

		std::cout << "Maximum Bad > 1: " << max_bad1 << " % (Frame: "
				<< max_bad1_id << ")" << std::endl;


		std::cout << "Minimum Bad > 3: " << min_bad3 << " % (Frame: "
				<< min_bad3_id << ")" <<std::endl;

		std::cout << "Maximum Bad > 3: " << max_bad3 << " % (Frame: "
				<< max_bad3_id << ")" << std::endl;
	}
    return 0;
}
