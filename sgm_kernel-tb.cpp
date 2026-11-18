#include "sgm_kernel-tb.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <algorithm>

int main()
{
	std::string left_dir = "/home/hamzy/SGM/Kitti-Data/training/image_2";
	std::string right_dir = "/home/hamzy/SGM/Kitti-Data/training/image_3";
	std::string gt_dir = "/home/hamzy/SGM/Kitti-Data/training/disp_noc_0";

    const int NUM_FRAME = 200;

    const int cy = WIN >> 1;

    int valid_frame_count = 0;

    double total_d1 = 0.0;
    double total_mae = 0.0;
    double total_bad3 = 0.0;

    double d1_per_frame[NUM_FRAME];
    double mae_per_frame[NUM_FRAME];
    double bad3_per_frame[NUM_FRAME];

    for(int i = 0; i < NUM_FRAME; ++i)
    {
        std::ostringstream ss;
        ss << std::setw(6) << std::setfill('0') << i;

        std::string id = ss.str();
        std::string file_name = id + "_10.png";

        std::string left_path = left_dir + "/" + file_name;
        std::string right_path = right_dir + "/" + file_name;
        std::string gt_path = gt_dir + "/" + file_name;

        /* Load as grayscale */
        cv::Mat left  = cv::imread(left_path,  cv::IMREAD_GRAYSCALE);
        cv::Mat right = cv::imread(right_path, cv::IMREAD_GRAYSCALE);
        cv::Mat gt = cv::imread(gt_path, cv::IMREAD_UNCHANGED);
        if (left.empty() || right.empty() || gt.empty())
        {
            std::cerr << "ERROR: Could not load input images:\n  "
                      << left_path << "\n  " << right_path <<  "\n  " << gt_path << std::endl;
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

        /* Resize the ground truth */
        const float scale_x = static_cast<float>(IMG_W) / static_cast<float>(gt.cols);

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
        static pix_t left_arr[IMG_H][IMG_W];
        static pix_t right_arr[IMG_H][IMG_W];
        static disp_t disp_arr[IMG_H][IMG_W];

        for(int r = 0; r < IMG_H; ++r)
        {
            const uint8_t* lp = left.ptr<uint8_t>(r);
            const uint8_t* rp = right.ptr<uint8_t>(r);

            for(int c = 0; c < IMG_W; ++c)
            {
                left_arr[r][c] = static_cast<pix_t>(lp[c]);
                right_arr[r][c] = static_cast<pix_t>(rp[c]);

                disp_arr[r][c] = 0;
            }
        }

        /* Run kernel */
        sgm_kernel(left_arr, right_arr, disp_arr);

        /* Retrieve output disparity */
        cv::Mat disp(IMG_H, IMG_W, CV_8U, cv::Scalar(0));
        using out_u_t = uint8_t;

        for(int r = 0; r < IMG_H; ++r)
        {
            int out_r = r - cy;

            if(out_r < 0)
                continue;

            for(int c = 0; c < IMG_W; ++c)
            {
                disp.at<out_u_t>(out_r, c) = static_cast<out_u_t>(disp_arr[r][c]);
            }
        }

        /* Accuracy metrics*/
        int kitti_valid_count = 0;
        int kitti_invalid_count = 0;

        int kitti_d1_err = 0;
        int bad3 = 0;

        double sum_abs_err = 0.0;

        cv::Mat err_map(IMG_H, IMG_W, CV_32F, cv::Scalar(0));

        for(int r = 0; r < IMG_H; ++r)
        {
            for(int c = 0; c < IMG_W; ++c)
            {
                float gt_disp = gt_f.at<float>(r,c);

                float est_disp = float(disp.at<out_u_t>(r,c));

                /* For all valid GT pixels */
                if(gt_disp <= 0.0f)
                {
                    ++kitti_invalid_count;
                    continue;
                }
                ++kitti_valid_count;

                float err_reduced = std::abs(est_disp - gt_disp);

                float err_orig = err_reduced / scale_x;

                err_map.at<float>(r,c) = err_orig;

                sum_abs_err += err_orig;

                if(err_orig > 3.0f)
                {
                    ++bad3;
                }

                bool abs_error_gt3 = err_orig > 3.0f;

                bool rel_error_gt5 = (err_reduced / gt_disp) > 0.05f;

                if(abs_error_gt3 && rel_error_gt5)
                {
                    ++kitti_d1_err;
                }
            }
        }
        if(kitti_valid_count == 0)
        {
            std::cerr << "ERROR: No KITTI-valid pixels at frame " << id
                << std::endl;
            return 6;
        }

        if((kitti_valid_count + kitti_invalid_count) != (IMG_H * IMG_W))
        {
            std::cerr << "ERROR: GT pixel count mismatch at frame " << id
                << std::endl;
            return 7;
        }

        double current_d1 = 100.0 * kitti_d1_err / kitti_valid_count;
        double current_mae = sum_abs_err / kitti_valid_count;
        double current_bad3 = 100.0 * bad3 / kitti_valid_count;

        d1_per_frame[i] = current_d1;

        mae_per_frame[i] = current_mae;

        bad3_per_frame[i] = current_bad3;

        total_d1 += current_d1;

        total_mae += current_mae;

        total_bad3 += current_bad3;

        ++valid_frame_count;

        std::cout << "Frame " << id << ", D1 = " << current_d1 << " %, MAE = "
            << current_mae << " px, Bad >3 = " << current_bad3 << " %"
			<< std::endl;

        if(i == 0)
        {
        #if (DISP > 256)
            cv::imwrite("disp_u16_frame0.png", disp);
        #else
            cv::imwrite("disp_u8_frame0.png", disp);
        #endif
            double disp_min = 0.0;
            double disp_max = 0.0;

            cv::minMaxLoc(disp, &disp_min, &disp_max);

            double scale_den = (disp_max > 0.0) ? disp_max :
            		std::max(1, DISP - 1);

            cv::Mat disp_vis;

            disp.convertTo(disp_vis, CV_8U, 255.0 / scale_den);

            cv::imwrite("disp_vis_frame0.png", disp_vis);

            cv::Mat disp_color;

            cv::applyColorMap(disp_vis, disp_color, cv::COLORMAP_JET);

            cv::imwrite("disp_color_frame0.png", disp_color);

            double err_max = 0.0;

            cv::minMaxLoc(err_map, nullptr, &err_max);

            cv::Mat err_vis;

            double err_scale = (err_max > 0.0) ? (255.0 / err_max) : 1.0;

            err_map.convertTo(err_vis, CV_8U, err_scale);

            cv::imwrite("err_map_frame0.png", err_vis);
         }
    }

    /* Dataset statistics */
    if(valid_frame_count == 0)
    {
        std::cerr << "ERROR: No valid frames evaluated." << std::endl;
        return 8;
    }

    double mean_d1 = total_d1 / valid_frame_count;

    double mean_mae = total_mae / valid_frame_count;

    double mean_bad3 = total_bad3 / valid_frame_count;

    double d1_squared_diff_sum = 0.0;
    double mae_squared_diff_sum = 0.0;
    double bad3_squared_diff_sum = 0.0;

    for(int i = 0; i < NUM_FRAME; ++i)
    {
        double d1_diff = d1_per_frame[i] - mean_d1;
        double mae_diff = mae_per_frame[i] - mean_mae;
        double bad3_diff = bad3_per_frame[i] - mean_bad3;

        d1_squared_diff_sum += d1_diff * d1_diff;
        mae_squared_diff_sum += mae_diff * mae_diff;
        bad3_squared_diff_sum +=bad3_diff * bad3_diff;
    }

    double std_d1 = std::sqrt(d1_squared_diff_sum / valid_frame_count);
    double std_mae = std::sqrt(mae_squared_diff_sum / valid_frame_count);
    double std_bad3 = std::sqrt(bad3_squared_diff_sum / valid_frame_count);

    /* Final results*/

    std::cout << "----------200-Frame Accuracy Evaluation----------\n";

    std::cout << "Mean D1: " << mean_d1 << " %\n";
    std::cout << "D1 Standard Deviation: " << std_d1 << " %\n";

    std::cout << "Mean MAE: " << mean_mae << " px\n";
    std::cout << "MAE Standard Deviation: " << std_mae << " px\n";

    std::cout << "Mean Bad >3: " << mean_bad3<< " %\n";
    std::cout << "Bad >3 Standard Deviation: " << std_bad3 << " %\n";

    return 0;
}
