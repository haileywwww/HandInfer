#include <fstream>
#include <vector>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std;
// 定义点云类型
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

// 相机内参
const double camera_factor = 1.0;
const double camera_cx = 939.2507269638458638;//563.788940;
const double camera_cy = 528.8425773400273329;//656.947067;
const double camera_fx = 1274.390077454887887;//1410.719299;
const double camera_fy = 1309.656335641865098;//1410.719299;

int depth_width = 896;//1920;//1080;
int depth_height = 768;//1080;//1280;
int sparse_rate = 2;

PointCloudT::Ptr depth2points(cv::Mat depth, PointCloudT::Ptr& cloud) {

    for (int m = 0; m < depth.rows; m+=sparse_rate) {
        for (int n = 0; n < depth.cols; n+=sparse_rate) {            
            float d = depth.at<float>(m, n);
            if (d == 0 || d > 0.600)// || m > data.size()-20)
                continue;

            PointT p;
            p.z = double(d) / camera_factor;
            p.x = (n - camera_cx) * p.z / camera_fx;
            p.y = (m - camera_cy) * p.z / camera_fy;
    
            cloud->points[m/sparse_rate * depth.cols/sparse_rate + n/sparse_rate] = p;
            // cloud->points.push_back(p);
        }
    }


    PointT p;
    p.z = 0 / camera_factor;
    p.x = 0;
    p.y = 0;
    cloud->points.push_back(p);
    p.z = 0.600 / camera_factor;
    p.x = 0;
    p.y = 0;
    cloud->points.push_back(p);



    return cloud;
}

std::vector<std::string> load_class_list()
{
    std::vector<std::string> class_list;
    std::ifstream ifs("config_files/classes.txt");
    std::string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    return class_list;
}

void load_net(cv::dnn::Net &net, bool is_cuda, const cv::String& model_path){
    // auto result = cv::dnn::readNet("config_files/hand.onnx");
    auto result = cv::dnn::readNet(model_path);
    if (is_cuda)
    {
        std::cout << "Attempty to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else
    {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
const float CONFIDENCE_THRESHOLD = 0.4;

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

cv::Mat format_yolov5(const cv::Mat &source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

void detect(cv::Mat &image, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className) {
    cv::Mat blob;

    auto input_image = format_yolov5(image);
    auto size = input_image.size();
    std::cout<<"yolo mat: "<<size.height<<" "<<size.width<<" "<<input_image.channels()<<"\n";
    cv::dnn::blobFromImage(input_image, blob, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    auto start = std::chrono::high_resolution_clock::now();
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    // std::cout << "===> Detect Time: " << diff.count() << " s\n";
    auto output_size = outputs[0].size();
    // std::cout<<"net_det output shape: "<< outputs.size() << " " << output_size.height << " " << output_size.width <<" " << outputs[0].channels() <<"\n";
    // (1,1,25200,1)

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    
    float *data = (float *)outputs[0].data;

    // const int dimensions = 85;// 80 classes + (x,y,w,h) + conf
    const int dimensions = 6;
    const int rows = 25200;
    
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {

        float confidence = data[4];    
        if (confidence >= CONFIDENCE_THRESHOLD) {
            // std::cout<<confidence<<"\n";
            float * classes_scores = data + 5;
            cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD) {

                confidences.push_back(confidence);

                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // std::cout<<left<<" "<<top<<" "<<width<<" "<<height<<"\n";
                boxes.push_back(cv::Rect(left, top, width, height));
            }

        }

        data += dimensions;

    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
        std::cout << "======================" <<std::endl;
        std::cout << confidences[idx] << " " << boxes[idx].x << " " << boxes[idx].y << " " <<boxes[idx].height << " " <<boxes[idx].width << std::endl;
        std::cout << "======================" <<std::endl;

    }
}

// adding gamma
cv::Mat adjust_gamma(const cv::Mat& image, double gamma) {
    // 创建一个与原图像相同大小的空图像
    cv::Mat new_image = cv::Mat::zeros(image.size(), image.type());

    // 遍历图像的每个像素
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            // 获取原图像的像素值
            int pixel = image.at<uchar>(y, x);

            // 计算新的像素值
            double new_pixel = cv::saturate_cast<uchar>(pow((double)pixel / 255.0, gamma) * 255);

            // 设置新图像的像素值
            new_image.at<uchar>(y, x) = (uchar)new_pixel;
        }
    }

    return new_image;
}

cv::Mat adjust_gamma_lut(const cv::Mat& image, double gamma) {
    CV_Assert(gamma >= 0);

    cv::Mat lookUpTable(1, 256, CV_8UC1);
    uchar* p = lookUpTable.ptr();
    for (int i = 0; i < 256; ++i) {
        p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    }

    cv::Mat res = image.clone();
    cv::LUT(image, lookUpTable, res);

    return res;
}

// padding 
cv::Mat pad_image(const cv::Mat& arr, const std::vector<int>& target_size, std::vector<int>& pad_size) {
    // 计算需要填充的行数和列数
    int rows_to_pad = target_size[0] - arr.rows; // height
    int cols_to_pad = target_size[1] - arr.cols; // width 

    // 计算每一侧需要填充的行数和列数
    int rows_to_pad_left = rows_to_pad / 2;
    int rows_to_pad_right = rows_to_pad - rows_to_pad_left;
    int cols_to_pad_top = cols_to_pad / 2;
    int cols_to_pad_bottom = cols_to_pad - cols_to_pad_top;

    // 使用cv::copyMakeBorder函数填充数组
    cv::Mat padded_arr;
    cv::copyMakeBorder(arr, padded_arr, rows_to_pad_left, rows_to_pad_right, cols_to_pad_top, cols_to_pad_bottom, cv::BORDER_CONSTANT);

    // 返回填充后的图像和填充的大小
    pad_size = {rows_to_pad_left, rows_to_pad_right, cols_to_pad_top, cols_to_pad_bottom};
    return padded_arr;
}

// depth predict 
cv::Mat depth_predict(cv::Mat& input, cv::dnn::Net &net_depth){
    // gamma 
    double gamma_value = 1/1.8;
    std::vector<int> target_size = {768,896};
    //cv::imwrite("input_22.png", input);
    cv::Mat input_gamma = adjust_gamma_lut(input, gamma_value);
    //cv::imwrite("input_gamma.png", input_gamma);
    // pad
    std::vector<int> pad_size;
    cv::Mat input_pad = pad_image(input_gamma, target_size, pad_size);
    //cv::imwrite("input_pad.png", input_pad);
    cv::Size size = input_pad.size();
    //std::cout<<"pad size: "<< size.height << " " << size.width << " " << input_pad.channels() << "\n";
    // model infer 
    cv::Mat blob;
    cv::Mat input_blob = input_pad.clone();
    //cv::imwrite("input_blob.png", input_blob);
    cv::dnn::blobFromImage(input_blob, blob, 1., cv::Size(896, 768), cv::Scalar(), true, false); // cv::Size(width, height) 
    // cv::dnn::blobFromImage(input_blob, blob, 1.);    
    net_depth.setInput(blob); // (b, c, h, w) 
    std::vector<cv::Mat> outputs;
    // net_depth.forward();
    auto start = std::chrono::high_resolution_clock::now();
    net_depth.forward(outputs, net_depth.getUnconnectedOutLayersNames());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    // std::cout << "===> Depth Time: " << diff.count() << " s\n";
    //std::cout<<"==========> TEST\n";
    auto output_size = outputs[0].size();
    //std::cout<<"net_depth output shape: "<< outputs.size() << " " << output_size.height << " " << output_size.width <<" " << outputs[0].channels() <<"\n"; // (1,1,1,1)
    float *data = (float *)outputs[0].data;

    /// save txt 
    // 创建一个文件流
    // std::ofstream file("output.txt");
    // // 检查文件是否成功打开
    // if (!file.is_open()) {
    //     std::cerr << "无法打开文件" << std::endl;
    //     return;
    // }
    // 遍历数据并写入文件
    int rows=768, cols=896;
    cv::Mat result(rows, cols, CV_32F);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result.at<float>(i, j) = (float)(1.0/data[i*cols+j]);
        }
    }
    // cut padding data
    cv::Rect padding_box;
    cv::Mat padding_result;
    // if(pad_size.size()>=4){
    padding_box.y = pad_size[0];
    padding_box.height = 768 - pad_size[1] - padding_box.x;
    padding_box.x = pad_size[2];
    padding_box.width = 896 - pad_size[3] - padding_box.y;   
    // std::cout<<"==> result:"<<result.rows<<" "<<result.cols<<"\n"; 
    std::cout<<"==> padding_box:"<<padding_box.x<<" "<<padding_box.y<<" "<<padding_box.width<<" "<<padding_box.height<<"\n";
    result(padding_box).copyTo(padding_result);

    // }
    return padding_result;
    // for (int i = 0; i < rows; ++i) {
    //     for (int j = 0; j < cols; ++j) {
    //         float depth_val = 1.0 / data[i * cols + j];
    //         file << depth_val;
    //         if (j < 896 - 1) {
    //             file << " "; // 在列之间添加空格
    //         }
    //     }
    //     file << "\n"; // 在行之间添加换行符
    // }
    // 关闭文件
    // file.close();
}

int main(int argc, char **argv){

    std::vector<std::string> class_list = load_class_list();

    // cap 
    cv::VideoCapture cap;

    // 打开视频捕获设备
    // 参数2是API后端，这里使用CAP_V4L2来指定V4L2后端
    // 参数3是设备索引，这里使用2，通常是0代表默认摄像头，1代表第二个摄像头，以此类推
    if (!cap.open(2, cv::CAP_V4L2)) {
        std::cerr << "Error: Unable to open video capture device." << std::endl;
        return -1;
    }

    // 设置视频捕获属性
    // 设置帧宽度
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    // 设置帧高度
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);

    // 检查是否成功设置属性
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to set video capture properties." << std::endl;
        return -1;
    }

    // 从摄像头捕获帧
    // cv::Mat input;
    // while (true) {
    //     // 从视频捕获设备读取帧
    //     cap >> input;
    //     cv::imshow("frame", input);
    //     cv::waitKey(1);

    // }

    // cv::Mat frame;
    // frame = cv::imread("hand.png");
    // auto fs = frame.size();
    // std::cout<<"frame shape: "<<fs.height<<" "<<fs.width<<" "<<frame.channels()<<"\n";

    bool is_cuda = true;
    cv::dnn::Net net_det, det_depth;
    load_net(net_det, is_cuda, "config_files/hand_det.onnx");
    // load_net(det_depth, is_cuda, "config_files/hand_depth.onnx");
    load_net(det_depth, is_cuda, "config_files/hand_mask_0125.onnx");

    // count time 
    auto start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    float fps = -1;
    int total_frames = 0;



    cv::Mat EmptyMat;
    PointCloudT::Ptr cloud1(new PointCloudT);
    cloud1->points.resize(depth_width * depth_height/sparse_rate/sparse_rate);
    depth2points(EmptyMat, cloud1);

    // Visualization
    pcl::visualization::PCLVisualizer viewer ("PointCloud");
    // pcl::visualization::PointCloudColorHandlerCustom<PointT> single_color (cloud1, (int) 255 * 1, (int) 255 * 1,
    //                                                                            (int) 255 * 1);
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> single_color(cloud1, "z");/// deep different color      
    viewer.addPointCloud (cloud1, single_color, "cloud_1_v1");
    viewer.setBackgroundColor (0, 0, 0);
    viewer.setCameraPosition (0, 0, -1, 0, 0, 0, 0, -1, 0);
    // viewer.setCameraPosition (-2, 0, 0.6, 4, -0, 0, 0, -1, 0);
    // viewer.setCameraPosition (0, -1, -1.5, 0, -0, 1, 0, -1, 0);
    viewer.setSize (1920, 1080);





    while(true){
    	// cv::Mat input;
    	cv::Mat frame;
        cap >> frame;  	
    	// auto fss = input.size();
    	// std::cout<<"input: "<<fss.height<<" "<<fss.width<<" "<<input.channels()<<"\n";
    	// cv::cvtColor(in2, frame, cv::COLOR_GRAY2RGB);
    	// cv::repeat(input, 3, 2, frame); // (1080,1920) -> (1080,1920,3)
    	auto fss = frame.size();
    	std::cout<<"frame: "<<fss.height<<" "<<fss.width<<" "<<frame.channels()<<"\n";
    	if (frame.empty()){
    	    std::cout << "frame is empty! \n";
    	}

    	auto start = std::chrono::high_resolution_clock::now();
    	std::vector<Detection> output;
    	detect(frame, net_det, output, class_list);
    	auto end = std::chrono::high_resolution_clock::now();
    	std::chrono::duration<double> diff = end-start;
    	std::cout << "detect Time: " << diff.count() << " s\n";

    	frame_count++;
    	total_frames++;

    	int detections = output.size();
    	std::cout<<"outputsize: "<<detections<<"\n";
    	std::vector<cv::Mat> boxes;
    	for (int i = 0; i < detections; ++i){
    	    auto detection = output[i];
    	    auto box = detection.box;
    	    auto classId = detection.class_id;
    	    const auto color = colors[classId % colors.size()];
    	    /// TODO depth predict model input max is (896,768), need optimized
            if(box.width>=896||box.height>=768||box.x<0||box.y<0||box.x+box.width>=frame.cols||box.y+box.height>=frame.rows){
                continue;
            }
            std::cout<<"==> detect_box:"<<box.x<<" "<<box.y<<" "<<box.width<<" "<<box.height<<"\n";

        start = std::chrono::high_resolution_clock::now();
    	    cv::Mat box_mat = frame(box);
    	    // cv::imwrite("box_mat.png", box_mat);

    	    cv::Mat result = depth_predict(box_mat, det_depth);
    	    cv::Mat out11_scaled;
    	    result.convertTo(out11_scaled, CV_8U, 200.0, 0.0); // 将图像数据转换为8位无符号整数

    	    // 应用颜色映射
    	    cv::Mat out11_colormap;
    	    cv::applyColorMap(out11_scaled, out11_colormap, cv::COLORMAP_MAGMA);

    	    // 显示结果图像
    	    cv::imshow("Colormap Image", out11_colormap);
    	    cv::waitKey(1);
    	    // boxes.push_back(frame(box));
    	    // cv::rectangle(frame, box, color, 3);
    	    // cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
    	    // cv::putText(frame, class_list[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

        end = std::chrono::high_resolution_clock::now();
    	diff = end-start;
    	std::cout << "process Time: " << diff.count() << " s\n";



        start = std::chrono::high_resolution_clock::now();
            viewer.spinOnce (1);
        end = std::chrono::high_resolution_clock::now();
    	diff = end-start;
    	std::cout << "cloud1 Time: " << diff.count() << " s\n";
        start = std::chrono::high_resolution_clock::now();
            
            PointCloudT::Ptr cloud2(new PointCloudT);
            cloud2->points.resize(depth_width * depth_height/sparse_rate/sparse_rate);  
        end = std::chrono::high_resolution_clock::now();
    	diff = end-start;
    	std::cout << "cloud2 Time: " << diff.count() << " s\n";
        start = std::chrono::high_resolution_clock::now();
            depth2points(result, cloud2);
        end = std::chrono::high_resolution_clock::now();
    	diff = end-start;
    	std::cout << "cloud3 Time: " << diff.count() << " s\n";
        start = std::chrono::high_resolution_clock::now();

            // pcl::visualization::PointCloudColorHandlerCustom<PointT> single_color2 (cloud1, (int) 255 * 1, (int) 255 * 1,
            //                                                                (int) 255 * 1);
            pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> single_color2(cloud2, "z");/// deep different color    

            viewer.updatePointCloud (cloud2, single_color2, "cloud_1_v1");
        end = std::chrono::high_resolution_clock::now();
    	diff = end-start;
    	std::cout << "cloud4 Time: " << diff.count() << " s\n" << std::endl;;

    	}
    }
    // cv::imwrite("output-hand.png", frame);
    // for(int i=0;i<boxes.size();i++){
    //     cv::imwrite("box.png", boxes[i]);
    // }

    return 0;
}
