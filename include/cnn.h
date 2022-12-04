#ifndef CNN_H
#define CNN_H

#include <vector>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

using vector3i = vector<vector<vector<int>>>;
using vector2i = vector<vector<int>>;
using vector1i = vector<int>;
using vector2f = vector<vector<float>>;
using vector1f = vector<float>;

class cnn {
private:
    int height, width;
    vector<vector3i> filters;
    vector1i filters_shift;
    vector<vector2f> w;
    vector1f w_shift;

    void init_filters();
    void init_w();
    void init_input(const Mat &image, vector3i &input);

    void convolution(const vector3i &input, vector3i filter, int filter_shift, vector2i &output);
    void ReLU(vector2i &inoutput);
    void max_pooling(vector2i &inoutput);
    void pixel_soft_max(const vector<vector2i> &input, vector2f &output);

    void fully_connected_layer(const vector2f &input, vector1f &class_img_predict);

public:
    cnn(int height, int width);

    void fit(const Mat &image, vector1f &class_img_predict);
};

#endif // CNN_H
