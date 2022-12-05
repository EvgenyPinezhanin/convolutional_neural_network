#include <cnn.h>

#include <random>
#include <cmath>

const int count_filters = 3;
const int size_filter = 3;
const int size_bgr = 3;
const int count_channel = 3;

void cnn::init_filters() {
    random_device dev;
    mt19937 gen(dev());

    for (int i = 0; i < count_filters; i++) {
        for (int j = 0; j < size_filter; j++) {
            for (int k = 0; k < size_filter; k++) {
                for (int l = 0; l < size_bgr; l++) {
                    filters[i][j][k][l] = 2.0 * (gen() - gen.min()) / (gen.max() - gen.min()) - 1.0;
                }
            }
        }
    }
    for (int i = 0; i < count_filters; i++) {
        filters_shift[i] = 2.0 * (gen() - gen.min()) / (gen.max() - gen.min()) - 1.0;
    }
}

void cnn::init_w() {
    random_device dev;
    mt19937 gen(dev());

    for (int i = 0; i < count_channel; i++) {
        for (int j = 0; j < (height - 2 * (size_filter / 2)) / 2; j++) {
            for (int k = 0; k < (width - 2 * (size_filter / 2)) / 2; k++) {
                w[i][j][k] = 2.0 * (gen() - gen.min()) / (gen.max() - gen.min()) - 1.0; 
            }
        }
    }
    for (int i = 0; i < count_channel; i++) {
        w_shift[i] = 2.0 * (gen() - gen.min()) / (gen.max() - gen.min()) - 1.0;
    }
}

void cnn::init_input(const Mat &image, vector3f &input) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < size_bgr; k++) {
                input[i][j][k] = (float)image.at<Vec3b>(i, j)[k];  
            }
        }
    }
}

void cnn::convolution(const vector3f &input, vector3f filter, float filter_shift, vector2f &output) {
    float sum;
    for (int i = 0; i < height - 2 * (size_filter / 2); i++) {
        for (int j = 0; j < width - 2 * (size_filter / 2); j++) {
            sum = 0.0;
            for (int x = i; x < i + size_filter; x++) {
                for (int y = j; y < j + size_filter; y++) {
                    for (int z = 0; z < size_bgr; z++) {
                        sum += input[x][y][z] * filter[x - i][y - j][z];
                    }
                }
            }
            output[i][j] = sum + filter_shift;
        }
    }
}

void cnn::ReLU(vector2f &inoutput) {
    for (int i = 0; i < inoutput.size(); i++) {
        for (int j = 0; j < inoutput[0].size(); j++) {
            inoutput[i][j] = max(inoutput[i][j], 0.f);
        }
    }
}

void cnn::max_pooling(vector2f &inoutput) {
    vector2f output(inoutput.size() / 2);
    for (int i = 0; i < inoutput.size() / 2; i++) {
        output[i].resize(inoutput[0].size() / 2);
    }
    for (int i = 0; i < inoutput.size() / 2; i++) {
        for (int j = 0; j < inoutput[0].size() / 2; j++) {
            output[i][j] = max( {inoutput[2 * i][2 * j], 
                                 inoutput[2 * i + 1][2 * j],
                                 inoutput[2 * i][2 * j + 1],
                                 inoutput[2 * i + 1][2 * j + 1] });
        }
    }
    inoutput = output;
}

void cnn::pixel_soft_max(vector<vector2f> &input) {
    float sum;
    for (int i = 0; i <input[0].size(); i++) {
        for (int j = 0; j < input[0][0].size(); j++) {
            sum = 0.f;
            for (int k = 0; k < size_bgr; k++) {
                sum += exp(input[k][i][j]);
            }
            for (int k = 0; k < size_bgr; k++) {
                input[k][i][j] = exp(input[k][i][j]) / sum;
            }
        }
    }
}

cnn::cnn(int _height, int _width) 
    : height(_height), width(_width), filters(count_filters), filters_shift(count_filters), w(count_channel), w_shift(count_channel) {
    for (int i = 0; i < count_filters; i++) {
        filters[i].resize(size_filter);
        for (int j = 0; j < size_filter; j++) {
            filters[i][j].resize(size_filter);
            for (int k = 0; k < size_filter; k++) {
                filters[i][j][k].resize(size_bgr);
            }
        }
    }
    init_filters();
    for (int i = 0; i < count_channel; i++) {
        w[i].resize((height - 2 * (size_filter / 2)) / 2);
        for (int j = 0; j < (height - 2 * (size_filter / 2)) / 2; j++) {
            w[i][j].resize((width - 2 * (size_filter / 2)) / 2); 
        }
    }
    init_w();
}

void cnn::fit(const Mat &image) {
    vector3f input(height);
    for (int i = 0; i < height; i++) {
        input[i].resize(width);
        for (int j = 0; j < width; j++) {
            input[i][j].resize(size_bgr);
        }
    }
    init_input(image, input);

    vector<vector2f> outputs(count_filters);
    for (int i = 0; i < count_filters; i++) {
        outputs[i].resize(height - 2 * (size_filter / 2));
        for (int j = 0; j < height - 2 * (size_filter / 2); j++) {
            outputs[i][j].resize(width - 2 * (size_filter / 2));
        }
    }

    for (int i = 0; i < count_filters; i++) {
        convolution(input, filters[i], filters_shift[i], outputs[i]);
    }
 
    for (int i = 0; i < count_filters; i++) {
        ReLU(outputs[i]);
    }

    for (int i = 0; i < count_filters; i++) {
        max_pooling(outputs[i]);
    }

    pixel_soft_max(outputs);
    
    cout << "Количество каналов " << outputs.size() << endl;
}