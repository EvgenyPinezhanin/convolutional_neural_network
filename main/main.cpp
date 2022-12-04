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
    vector<float> class_img_predict(3);

    namedWindow("convolutional neural network", WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);
    resizeWindow("convolutional neural network", 900, 900 * src.rows / src.cols);

    cnn_model.fit(src, class_img_predict);
    
    string class_name = "Is this a "; 
    if (class_img_predict[0] > class_img_predict[1]) {
        if (class_img_predict[0] > class_img_predict[2]) {
            class_name += "cat";
        } else {
            class_name += "bird";
        }
    } else {
        if (class_img_predict[1] > class_img_predict[2]) {
            class_name += "dog";
        } else {
            class_name += "bird";
        }
    }
    putText(src, class_name, Point(30, 60), 1, 4, (255,255,255), 5);

    imshow("convolutional neural network", src);

    waitKey(0);
    return 0;
}