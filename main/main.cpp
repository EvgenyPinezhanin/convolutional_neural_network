#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cnn.h>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    String image_path = samples::findFile("resource/cat.jpg");
    Mat src = imread(image_path, IMREAD_COLOR);
    cnn cnn_model(src.rows, src.cols);

    namedWindow("convolutional neural network", WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);
    resizeWindow("convolutional neural network", 900, 900 * src.rows / src.cols);

    cnn_model.fit(src);

    imshow("convolutional neural network", src);

    waitKey(0);
    return 0;
}