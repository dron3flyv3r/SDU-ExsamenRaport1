#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

void linearFilter(const Mat &src, const Mat &kern, Mat &out) {
    out = Mat::zeros(src.size(), src.type());
    int ksize = kern.rows;
    int kcenter = ksize / 2;
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            float sum = 0.0;
            for (int m = 0; m < ksize; ++m) {
                for (int n = 0; n < ksize; ++n) {
                    int ii = i + m - kcenter;
                    int jj = j + n - kcenter;
                    if (ii >= 0 && ii < src.rows && jj >= 0 && jj < src.cols) {
                        sum += src.at<uchar>(ii, jj) * kern.at<float>(ksize - 1 - m, ksize - 1 - n);
                    }
                }
            }
            out.at<uchar>(i, j) = static_cast<uchar>(sum);
        }
    }
}

int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cout << "Usage: ./main <imagefile.jpg/png>"<< std::endl;
        return -1;
    }
    std::string filename = argv[1];

    Mat src = imread(filename, IMREAD_GRAYSCALE);
    imshow("src", src);
    waitKey(0);
    CV_Assert(src.type() == CV_8UC1);

    // Create uniform kernel
    Mat kernel(3,3, CV_32FC1, Scalar(1.0/9.0));
    CV_Assert(kernel.type() == CV_32FC1);

    // Apply linear filter
    Mat output;
    linearFilter(src, kernel, output);
    namedWindow("Linear filter output");
    imshow("Linear filter output", output);
    waitKey(0);

    // Test with custom kernel on impulse image
    Mat impulse = Mat::zeros(5, 5, CV_8UC1);
    impulse.at<uchar>(2, 2) = 1;

    Mat customKernel = (cv::Mat_<float>(3,3) << 1,2,3,4,5,6,7,8,9);
    linearFilter(impulse,customKernel,output);

    for(int i=0; i<output.rows; i++) {
        for (int j = 0; j < output.cols; j++) { std::cout << static_cast<int>(output.at<uchar>(i, j)) << " "; }
        std::cout << std::endl;
    }

    return 0;
}
