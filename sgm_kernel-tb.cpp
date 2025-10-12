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

    std::string left_path  = (argc > 1) ? argv[1] :  "/home/hamzy/SGM/Kitti-Data/training/image_2/000000_10.png";
    std::string right_path = (argc > 2) ? argv[2] : "/home/hamzy/SGM/Kitti-Data/training/image_3/000000_10.png";

    if (!file_exist(left_path) || !file_exist(right_path))
    {
    	std::cerr << "ERROR: Provide left & right images" << std::endl;
    	return 1;
    }
    /* Load as grayscale */
    cv::Mat left  = cv::imread(left_path,  cv::IMREAD_GRAYSCALE);
    cv::Mat right = cv::imread(right_path, cv::IMREAD_GRAYSCALE);
    if (left.empty() || right.empty())
    {
        std::cerr << "ERROR: Could not load input images:\n  "
                  << left_path << "\n  " << right_path << std::endl;
        return 1;
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
    if (left.cols != IMG_W || left.rows != IMG_H ||
        right.cols != IMG_W || right.rows != IMG_H) {
        std::cerr << "ERROR: Size mismatch after resize. "
                  << "Expected (" << IMG_W << "x" << IMG_H << ").\n";
        return 1;
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
#if (DISP > 256)
    cv::Mat disp(IMG_H, IMG_W, CV_16U);
    using out_u_t = uint16_t;
#else
    cv::Mat disp(IMG_H, IMG_W, CV_8U);
        using out_u_t = uint8_t;
#endif

    const int expected = IMG_W * IMG_H;
    int result = 0;
    for (int r = 0; r < IMG_H; ++r)
    {
#if (DISP > 256)
        uint16_t *dp = disp.ptr<uint16_t>(r);
#else
        uint8_t *dp = disp.ptr<uint8_t>(r);
#endif
        for (int c = 0; c < IMG_W; ++c)
        {
        	if (disp_stream.empty())
        	{
        		std::cerr << "ERROR: disp_stream underrun at pixel "
        				<< result << "/" << expected << std::endl;
        		return 0;
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
        return 2;
    }

    /* Save result */
#if (DISP > 256)
    cv::imwrite("disp_u16.png", disp);
#else
    cv::imwrite("disp_u8.png", disp);
#endif
    std::cout << "OK: Disparity map written to disp.png ("
              << IMG_W << "x" << IMG_H << ")\n";

    double mn = 0, mx = 0;
        cv::minMaxLoc(disp, &mn, &mx);
        double scale_den = (mx > 0) ? mx : std::max(1, DISP - 1);
        cv::Mat disp_vis_8u;
        disp.convertTo(disp_vis_8u, CV_8U, 255.0 / scale_den);

        cv::imwrite("disp_vis.png", disp_vis_8u);

        cv::Mat disp_color;
        cv::applyColorMap(disp_vis_8u, disp_color, cv::COLORMAP_JET);
        cv::imwrite("disp_color.png", disp_color);

        std::cout << "disp min=" << mn << " max=" << mx << "\n";
        return 0;
}
