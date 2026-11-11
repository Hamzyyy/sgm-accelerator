import cv2
import sys
import os

IMG_W = 320
IMG_H = 96
NUM_FRAMES = 200

left_dir  = "/home/hamzy/SGM/Kitti-Data/training/image_2"
right_dir = "/home/hamzy/SGM/Kitti-Data/training/image_3"
disp_dir  = "/home/hamzy/SGM/Kitti-Data/training/disp_noc_0"


def write_header(img, array_name, filename):
    with open(filename, "w") as f:
        f.write("#pragma once\n")
        f.write("#include <cstdint>\n\n")
        f.write(
            f"static const uint8_t {array_name}[{IMG_H}][{IMG_W}] = {{\n"
        )

        for r in range(IMG_H):
            f.write("    {")
            row_vals = [
                str(int(img[r, c]))
                for c in range(IMG_W)
            ]
            f.write(", ".join(row_vals))
            f.write("}")

            if r != IMG_H - 1:
                f.write(",")

            f.write("\n")

        f.write("};\n")


def write_float_header(img, array_name, filename):
    with open(filename, "w") as f:
        f.write("#pragma once\n")
        f.write("#include <cstdint>\n\n")
        f.write(
            f"static const float {array_name}"
            f"[{IMG_H}][{IMG_W}] = {{\n"
        )

        for r in range(IMG_H):
            f.write("    {")
            row_vals = [
                str(float(img[r, c]))
                for c in range(IMG_W)
            ]
            f.write(", ".join(row_vals))
            f.write("}")

            if r != IMG_H - 1:
                f.write(",")

            f.write("\n")

        f.write("};\n")


def main():
    scale_x_values = []

    for i in range(NUM_FRAMES):
        file_name = f"{i:06d}_10.png"
        frame_name = f"{i:06d}_10"

        left_header = "left_" + frame_name + ".hpp"
        right_header = "right_" + frame_name + ".hpp"
        gt_header = "gt_" + frame_name + ".hpp"

        left_path = os.path.join(left_dir, file_name)
        right_path = os.path.join(right_dir, file_name)
        gt_path = os.path.join(disp_dir, file_name)

        left = cv2.imread(
            left_path,
            cv2.IMREAD_GRAYSCALE
        )

        right = cv2.imread(
            right_path,
            cv2.IMREAD_GRAYSCALE
        )

        gt = cv2.imread(
            gt_path,
            cv2.IMREAD_UNCHANGED
        )

        if left is None or right is None or gt is None:
            print("ERROR: Could not read input images")
            sys.exit(1)

        left = cv2.resize(
            left,
            (IMG_W, IMG_H),
            interpolation=cv2.INTER_AREA
        )

        right = cv2.resize(
            right,
            (IMG_W, IMG_H),
            interpolation=cv2.INTER_AREA
        )

        scale_x = IMG_W / gt.shape[1]
        scale_x_values.append(scale_x)

        gt_f = gt.astype("float32") / 256.0

        gt_f = cv2.resize(
            gt_f,
            (IMG_W, IMG_H),
            interpolation=cv2.INTER_NEAREST
        )

        gt_f = gt_f * scale_x

        write_header(
            left,
            "left_" + frame_name,
            left_header
        )

        write_header(
            right,
            "right_" + frame_name,
            right_header
        )

        write_float_header(
            gt_f,
            "gt_" + frame_name,
            gt_header
        )

        print("Generated", frame_name)

    # Generate scale_x_frames.hpp
    with open("scale_x_frames.hpp", "w") as f:
        f.write("#pragma once\n\n")

        f.write(
            f"static const float "
            f"scale_x_frames[{NUM_FRAMES}] = {{\n"
        )

        for i in range(NUM_FRAMES):
            f.write(f"    {scale_x_values[i]}f")

            if i != NUM_FRAMES - 1:
                f.write(",")

            f.write("\n")

        f.write("};\n")


    with open("dataset_frames.hpp", "w") as f:

        f.write("#pragma once\n")
        f.write('#include "sgm_config.hpp"\n\n')


        # Include all 200 left/right/GT frame headers
        for i in range(NUM_FRAMES):
            frame_name = f"{i:06d}_10"

            f.write(
                f'#include "dataset/'
                f'left_{frame_name}.hpp"\n'
            )

            f.write(
                f'#include "dataset/'
                f'right_{frame_name}.hpp"\n'
            )

            f.write(
                f'#include "dataset/'
                f'gt_{frame_name}.hpp"\n'
            )

            f.write("\n")

        f.write(
            '#include "dataset/scale_x_frames.hpp"\n\n'
        )

        # Create left frame
        f.write(
            "static const uint8_t "
            "(*left_frames[])[IMG_W] = {\n"
        )

        for i in range(NUM_FRAMES):
            frame_name = f"{i:06d}_10"

            f.write(
                f"    left_{frame_name}"
            )

            if i != NUM_FRAMES - 1:
                f.write(",")

            f.write("\n")

        f.write("};\n\n")

        # Create right frame
        f.write(
            "static const uint8_t "
            "(*right_frames[])[IMG_W] = {\n"
        )

        for i in range(NUM_FRAMES):
            frame_name = f"{i:06d}_10"

            f.write(
                f"    right_{frame_name}"
            )

            if i != NUM_FRAMES - 1:
                f.write(",")

            f.write("\n")

        f.write("};\n\n")

        # Create GT 
        f.write(
            "static const float "
            "(*gt_frames[])[IMG_W] = {\n"
        )

        for i in range(NUM_FRAMES):
            frame_name = f"{i:06d}_10"

            f.write(
                f"    gt_{frame_name}"
            )

            if i != NUM_FRAMES - 1:
                f.write(",")

            f.write("\n")

        f.write("};\n\n")

        # NUM_FRAMES 
        f.write(
            "constexpr int NUM_FRAMES = "
            "sizeof(left_frames) / "
            "sizeof(left_frames[0]);\n"
        )


    print(
        "Generated dataset_frames.hpp "
        f"for {NUM_FRAMES} frames"
    )


if __name__ == "__main__":
    main()
