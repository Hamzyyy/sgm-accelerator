#include "sgm_kernel-tb.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

static inline bool file_exist(const std::string &p)
{
	return !p.empty();
}

int main(int argc, char** argv)
{

    std::string left_path  = (argc > 1) ? argv[1] :
    		"/home/hamzy/SGM/Kitti-Data/training/image_2/000000_10.png";
    std::string right_path = (argc > 2) ? argv[2] :
    		"/home/hamzy/SGM/Kitti-Data/training/image_3/000000_10.png";

    std::string gt_path = (argc > 3) ? argv[3] :
        	"/home/hamzy/SGM/Kitti-Data/training/disp_noc_0/000000_10.png";

    if (!file_exist(left_path) || !file_exist(right_path) || !file_exist(gt_path))
    {
    	std::cerr << "ERROR: Provide left & right images and the ground truth"
    			<< " left: " << left_path << "\n"
				<< "right: " << right_path << "\n"
				<< "gt: " << gt_path << std::endl;
    	return 1;
    }
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

    /* Before resize the GT count valid/invalid pixels */
    int original_total_pixels = 0;
    int original_gt_valid_count = 0, original_gt_invalid_count = 0;

    for (int r = 0; r < gt.rows; ++r)
    {
    	for (int c = 0; c < gt.cols; ++c)
    	{
    		++original_total_pixels;
    		uint16_t raw_gt = gt.at<uint16_t>(r,c);
    		if(raw_gt <= 0)
    			{
    				++original_gt_invalid_count;
    				continue;
    			}
    		++original_gt_valid_count;
    	}
    }

    std::cout << "gt type = " << gt.type() << "\n";
    std::cout << "Pixel count on the original GT " << "\n";
    std::cout << "Total pixels: " << original_total_pixels << "\n";
    std::cout << "GT-valid pixels (>0): " << original_gt_valid_count << "\n";
    std::cout << "GT-invalid pixels (<=0): " << original_gt_invalid_count << "\n";

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

    /* Simulate AXI4-Stream interfaces */
    hls::stream<pix_t> left_stream;
    hls::stream<pix_t> right_stream;
    hls::stream<pix_t> disp_stream;

    for (int r = 0; r < IMG_H; ++r)
    {
        const uint8_t* lp = left.ptr<uint8_t>(r);
        const uint8_t* rp = right.ptr<uint8_t>(r);
        for (int c = 0; c < IMG_W; ++c)
        {
            left_stream.write(static_cast<pix_t>(lp[c]));
            right_stream.write(static_cast<pix_t>(rp[c]));
        }
    }

    /* Run kernel */
    sgm_kernel(left_stream, right_stream, disp_stream);

    /* Retrieve output disparity */
    cv::Mat disp(IMG_H, IMG_W, CV_8U);
    using out_u_t = uint8_t;

    const int expected = IMG_W * IMG_H;
    int result = 0;

    for (int r = 0; r < IMG_H; ++r)
    {

    uint8_t *dp = disp.ptr<uint8_t>(r);

        for (int c = 0; c < IMG_W; ++c)
        {
        	if (disp_stream.empty())
        	{
        		std::cerr << "ERROR: disp_stream underrun at pixel "
        				<< result << "/" << expected << std::endl;
        		return 4;
        	}
        	out_u_t v = static_cast<out_u_t>(disp_stream.read());
            dp[c] = v;
            result++;
        }
    }

    if (result != expected)
    {
        std::cerr << "ERROR: expected " << expected << " disparity pixels, resulted "
        		<< result << std::endl;
        return 5;
    }


    /* Run software SGM baseline */
    cv::Mat disp_sw;
    sgm_sw(left, right, disp_sw);

    cv::imwrite("disp_sw_u8.png", disp_sw);
    std::cout << "OK: Software disparity written to disp_sw_u8.png\n";

    /* Compare HW kernel output vs SW output */
    int diff_count = 0;
    for (int r = 0; r < IMG_H; ++r)
    {
        for (int c = 0; c < IMG_W; ++c)
        {
            int hw_v = int(disp.at<uint8_t>(r, c));
            int sw_v = int(disp_sw.at<uint8_t>(r, c));

            if (hw_v != sw_v)
                diff_count++;
        }
    }

    int diff_same = 0;
    int diff_sw_left = 0;
    int diff_sw_right = 0;

    for (int r = 0; r < IMG_H; ++r)
    {
        for (int c = 1; c < IMG_W - 1; ++c)
        {
            int hw = int(disp.at<uint8_t>(r, c));

            if (hw != int(disp_sw.at<uint8_t>(r, c)))
                diff_same++;

            if (hw != int(disp_sw.at<uint8_t>(r, c - 1)))
                diff_sw_left++;

            if (hw != int(disp_sw.at<uint8_t>(r, c + 1)))
                diff_sw_right++;
        }
    }

    std::cout << "Diff same    = " << diff_same << "\n";
    std::cout << "Diff sw c-1  = " << diff_sw_left << "\n";
    std::cout << "Diff sw c+1  = " << diff_sw_right << "\n";

    std::cout << "HW/SW different pixels = "
              << diff_count << " / " << IMG_W * IMG_H << "\n";

    for (int r = 0; r < IMG_H; ++r)
    {
        for (int c = 0; c < IMG_W; ++c)
        {
            int hw = int(disp.at<uint8_t>(r, c));
            int sw = int(disp_sw.at<uint8_t>(r, c));

            if (hw != sw)
            {
                std::cout << "First mismatch at r=" << r
                          << " c=" << c
                          << " HW=" << hw
                          << " SW=" << sw << "\n";
                goto done_mismatch_debug;
            }
        }
    }

    done_mismatch_debug:
	std::cout << "SW disp(48,80)  = " << int(disp_sw.at<uint8_t>(48,80)) << "\n";
	std::cout << "SW disp(48,160) = " << int(disp_sw.at<uint8_t>(48,160)) << "\n";
	std::cout << "SW disp(48,240) = " << int(disp_sw.at<uint8_t>(48,240)) << "\n";

    double disp_min = 0.0, disp_max = 0.0;
    cv::minMaxLoc(disp, &disp_min, &disp_max);
    std::cout << "disp min=" << disp_min << " max=" << disp_max << "\n";

    int nonzero = 0;
    for (int r = 0; r < IMG_H; ++r)
    {
        for (int c = 0; c < IMG_W; ++c)
        {
            if (disp.at<out_u_t>(r,c) != 0) nonzero++;
        }
    }
    std::cout << "nonzero disparity pixels: " << nonzero
              << " / " << (IMG_W * IMG_H) << "\n";

    int test_r = IMG_H / 2;
    for (int c : {IMG_W/4, IMG_W/2, 3*IMG_W/4})
    {
        std::cout << "disp(" << test_r << "," << c << ") = "
                  << int(disp.at<out_u_t>(test_r, c)) << "\n";
    }

    int count0 = 0, count1to3 = 0, count4plus = 0;
    for (int r = 0; r < IMG_H; ++r) {
        for (int c = 0; c < IMG_W; ++c) {
            int v = int(disp.at<out_u_t>(r,c));
            if (v == 0) count0++;
            else if (v <= 3) count1to3++;
            else count4plus++;
        }
    }
    std::cout << "count0=" << count0
              << " count1to3=" << count1to3
              << " count4plus=" << count4plus << "\n";

    /* Save result */
    const char* raw_name = "disp_u8.png";

    static_assert(DISP <= 256, "Current output saving assume 8-bit disparity");

    cv::imwrite(raw_name, disp);
    std::cout << "OK: Disparity map written to " << raw_name << " ("
            << IMG_W << "x" << IMG_H << ")\n";

    /* Visualization */
    double mn = 0, mx = 0;
        cv::minMaxLoc(disp, &mn, &mx);
        double scale_den = (mx > 0) ? mx : std::max(1, DISP - 1);
        cv::Mat disp_vis_8u;
        disp.convertTo(disp_vis_8u, CV_8U, 255.0 / scale_den);

        cv::imwrite("disp_vis.png", disp_vis_8u);
        cv::Mat disp_color;
        cv::applyColorMap(disp_vis_8u, disp_color, cv::COLORMAP_JET);
        cv::imwrite("disp_color.png", disp_color);

        int total_pixels = 0;
        int eval_valid_count = 0, eval_invalid_count = 0;
        int zero_count_on_gt_valid= 0;
        int nonzero_count_on_gt_valid= 0;
        int out_of_range_on_gt_valid= 0;

        int bad1 = 0, bad3 = 0;
        double sum_abs_err = 0.0;
        cv::Mat err_map(IMG_H, IMG_W, CV_32F, cv::Scalar(0));

        const int cx = WIN >> 1;

        const int valid_r_min = WIN - 1;
        const int valid_c_min = (DISP - 1) + cx;
        const int valid_c_max = IMG_W - cx;

        for (int r = 0; r < IMG_H; ++r)
        {
        	for (int c = 0; c < IMG_W; ++c)
        	{
        		++total_pixels;
        		float gt_disp = gt_f.at<float>(r,c);

                bool gt_valid = gt_disp > 0.0f;
                bool roi_valid =
                    (r >= valid_r_min) &&
                    (c >= valid_c_min) &&
                    (c < valid_c_max);

                bool disp_range_valid =
                    (gt_disp >= 0.0f) &&
                    (gt_disp < DISP);

                bool eval_valid = gt_valid && roi_valid && disp_range_valid;

        		if(!eval_valid)
        			{
        				++eval_invalid_count;
        				continue;
        			}
        		++eval_valid_count;

        		float est_disp = float(disp.at<out_u_t>(r,c));

        		if(est_disp == 0.0f)
        			++zero_count_on_gt_valid;
        		else
        			++nonzero_count_on_gt_valid;

        		if(est_disp < 0.0f || est_disp >= DISP)
        			++out_of_range_on_gt_valid;

        		float err = std::abs(est_disp - gt_disp);
        		err_map.at<float>(r,c) = err;

        		sum_abs_err += err;
        		if (err > 1.0f) bad1++;
        		if (err > 3.0f) bad3++;
        	}
        }

        double err_max = 0.0;
        cv::minMaxLoc(err_map, nullptr, &err_max);

        cv::Mat err_vis;
        double scale = (err_max > 0.0) ? (255.0 / err_max) : 1.0;

        err_map.convertTo(err_vis, CV_8U, scale);

        cv::imwrite("err_map.png", err_vis);

        std::cout << "Total pixels: " << total_pixels << "\n";
        std::cout << "Evaluation-valid pixels (>0): " << eval_valid_count << "\n";
        std::cout << "Excluded invalid pixels (<=0): " << eval_invalid_count << "\n";
        std::cout << "est == 0 on GT-valid pixels: " << zero_count_on_gt_valid << "\n";
        std::cout << "est != 0 on GT-valid pixels: " << nonzero_count_on_gt_valid << "\n";
        std::cout << "est out of range on GT-valid pixels: " << out_of_range_on_gt_valid << "\n";

        if (eval_valid_count == 0)
        {
            std::cerr << "ERROR: No valid GT pixels for comparison\n";
            return 6;
        }

        std::cout << "MAE: " << (sum_abs_err / eval_valid_count) << "\n";
        std::cout << "Bad >1 px: " << (100.0 * bad1 / eval_valid_count) << "%\n";
        std::cout << "Bad >3 px: " << (100.0 * bad3 / eval_valid_count) << "%\n";

        return 0;
}
