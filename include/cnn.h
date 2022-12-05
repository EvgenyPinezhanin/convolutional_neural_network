#ifndef CNN_H
#define CNN_H

#include <vector>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

using vector3f = vector<vector<vector<int>>>;
using vector2f = vector<vector<float>>;
using vector1f = vector<float>;

class cnn {
private:
    int height, width;
    vector<vector3f> filters;
    vector1f filters_shift;
    vector<vector2f> w;
    vector1f w_shift;

    void init_filters();
    void init_w();
    void init_input(const Mat &image, vector3f &input);

    void convolution(const vector3f &input, vector3f filter, float filter_shift, vector2f &output);
    void ReLU(vector2f &inoutput);
    void max_pooling(vector2f &inoutput);
    void pixel_soft_max(vector<vector2f> &input);

public:
    cnn(int height, int width);

    void fit(const Mat &image);
};

#endif // CNN_H
