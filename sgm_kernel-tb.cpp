#include "sgm_kernel-tb.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

int main() {
    /* --- Adjust paths to your image pairs --- */
    const std::string left_path  = "/home/hamzy/SGM/Kitti-Data/training/image_2/000000_10.png";
    const std::string right_path = "/home/hamzy/SGM/Kitti-Data/training/image_3/000000_10.png";

    /* Load as grayscale */
    cv::Mat left  = cv::imread(left_path,  cv::IMREAD_GRAYSCALE);
    cv::Mat right = cv::imread(right_path, cv::IMREAD_GRAYSCALE);
    if (left.empty() || right.empty()) {
        std::cerr << "ERROR: Could not load input images:\n  "
                  << left_path << "\n  " << right_path << std::endl;
        return 1;
    }

    /* Ensure size matches kernel compile-time shape */
    if (left.cols != IMG_W || left.rows != IMG_H) {
        cv::resize(left,  left,  cv::Size(IMG_W, IMG_H), 0, 0, cv::INTER_AREA);
    }
    if (right.cols != IMG_W || right.rows != IMG_H) {
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

    /* Push pixels row-major */
    for (int r = 0; r < IMG_H; ++r) {
        const uint8_t* lp = left.ptr<uint8_t>(r);
        const uint8_t* rp = right.ptr<uint8_t>(r);
        for (int c = 0; c < IMG_W; ++c) {
            left_stream.write(static_cast<pix_t>(lp[c]));
            right_stream.write(static_cast<pix_t>(rp[c]));
        }
    }

    /* Run kernel */
    sgm_kernel(left_stream, right_stream, disp_stream);

    /* Retrieve output disparity */
    cv::Mat disp(IMG_H, IMG_W, CV_8U);
    for (int r = 0; r < IMG_H; ++r) {
        uint8_t* dp = disp.ptr<uint8_t>(r);
        for (int c = 0; c < IMG_W; ++c) {
            dp[c] = static_cast<uint8_t>(disp_stream.read());
        }
    }

    /* Save result */
    cv::imwrite("disp.png", disp);
    std::cout << "OK: Disparity map written to disp.png ("
              << IMG_W << "x" << IMG_H << ")\n";

    cv::Mat disp_vis;
    disp.convertTo(disp_vis, CV_8U, 255.0 / (DISP - 1));
    cv::imwrite("disp_vis.png", disp_vis);

    double mn, mx;
    cv::minMaxLoc(disp, &mn, &mx);
    std::cout << "disp min=" << mn << " max=" << mx << "\n";

    cv::Mat disp_color;
    cv::applyColorMap(disp_vis, disp_color, cv::COLORMAP_JET);
    cv::imwrite("disp_color.png", disp_color);


    return 0;
}
