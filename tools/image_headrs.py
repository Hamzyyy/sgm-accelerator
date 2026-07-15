import cv2
import sys

IMG_W = 320
IMG_H = 96

def write_header(img, array_name, filename):
    with open(filename, "w") as f:
        f.write("#pragma once\n")
        f.write("#include <cstdint>\n\n")
        f.write(f"static const uint8_t {array_name}[{IMG_H}][{IMG_W}] = {{\n")

        for r in range(IMG_H):
            f.write("    {")
            row_vals = [str(int(img[r, c])) for c in range(IMG_W)]
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
        f.write(f"static const float {array_name}[{IMG_H}][{IMG_W}] = {{\n")

        for r in range(IMG_H):
            f.write("    {")
            row_vals = [str(float(img[r, c])) for c in range(IMG_W)]
            f.write(", ".join(row_vals))
            f.write("}")
            if r != IMG_H - 1:
                f.write(",")
            f.write("\n")

        f.write("};\n")        

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 make_image_headers.py left.png right.png gt.png")
        sys.exit(1)

    left_path = sys.argv[1]
    right_path = sys.argv[2]
    gt_path = sys.argv[3]

    left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

    if left is None or right is None or gt is None:
        print("ERROR: Could not read input images")
        sys.exit(1)

    left = cv2.resize(left, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    
    scale_x = IMG_W/gt.shape[1]
    
    gt_f = gt.astype("float32") / 256.0
    gt_f = cv2.resize(gt_f, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    gt_f = gt_f * scale_x

    write_header(left, "left_img", "left_img.hpp")
    write_header(right, "right_img", "right_img.hpp")
    write_float_header(gt_f, "gt_disp", "gt_disp.hpp")

    print("Generated left_img.hpp, right_img.hpp and gt_disp.hpp")

if __name__ == "__main__":
    main()
