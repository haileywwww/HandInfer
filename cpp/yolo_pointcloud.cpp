#include <fstream>
#include <vector>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std;
using namespace cv;
using namespace cv::dnn;

///////////////////////////////////////////////////////////////////////////////
/// hand segment
///////////////////////////////////////////////////////////////////////////////
int _netWidth = 640;   //ONNX图片输入宽度
int _netHeight = 640;  //ONNX图片输入高度
float _classThreshold = 0.25;
float _nmsThreshold = 0.45;
float _maskThreshold = 0.5;
std::vector<std::string> _className = {"handseg"};
struct MaskParams {
	//int segChannels = 32;
	//int segWidth = 160;
	//int segHeight = 160;
	int netWidth = 640;
	int netHeight = 640;
	float maskThreshold = 0.5;
	cv::Size srcImgShape;
	cv::Vec4d params;

};
struct OutputSeg {
	int id;             //结果类别id
	float confidence;   //结果置信度
	cv::Rect box;       //矩形框
	cv::Mat boxMask;       //矩形框内mask，节省内存空间和加快速度
};
void LetterBox(const cv::Mat& image, cv::Mat& outImage,
	cv::Vec4d& params, //[ratio_x,ratio_y,dw,dh]
	const cv::Size& newShape = cv::Size(640, 640),
	bool autoShape = false,
	bool scaleFill = false,
	bool scaleUp = true,
	int stride = 32,
	const cv::Scalar& color = cv::Scalar(114, 114, 114));

void DrawPred(cv::Mat& img,
	std::vector<OutputSeg> result,
	std::vector<std::string> classNames,
	std::vector<cv::Scalar> color,
	bool isVideo = false
);

void GetMask2(const Mat& maskProposals, const Mat& maskProtos, OutputSeg& output, const MaskParams& maskParams) {
	int net_width = maskParams.netWidth;
	int net_height = maskParams.netHeight;
	int seg_channels = maskProtos.size[1];
	int seg_height = maskProtos.size[2];
	int seg_width = maskProtos.size[3];
	float mask_threshold = maskParams.maskThreshold;
	Vec4f params = maskParams.params;
	Size src_img_shape = maskParams.srcImgShape;

	Rect temp_rect = output.box;
	//crop from mask_protos
	int rang_x = floor((temp_rect.x * params[0] + params[2]) / net_width * seg_width);
	int rang_y = floor((temp_rect.y * params[1] + params[3]) / net_height * seg_height);
	int rang_w = ceil(((temp_rect.x + temp_rect.width) * params[0] + params[2]) / net_width * seg_width) - rang_x;
	int rang_h = ceil(((temp_rect.y + temp_rect.height) * params[1] + params[3]) / net_height * seg_height) - rang_y;

	//如果下面的 mask_protos(roi_rangs).clone()位置报错，说明你的output.box数据不对，或者矩形框就1个像素的，开启下面的注释部分防止报错。
	rang_w = MAX(rang_w, 1);
	rang_h = MAX(rang_h, 1);
	if (rang_x + rang_w > seg_width) {
		if (seg_width - rang_x > 0)
			rang_w = seg_width - rang_x;
		else
			rang_x -= 1;
	}
	if (rang_y + rang_h > seg_height) {
		if (seg_height - rang_y > 0)
			rang_h = seg_height - rang_y;
		else
			rang_y -= 1;
	}

	vector<Range> roi_rangs;
	roi_rangs.push_back(Range(0, 1));
	roi_rangs.push_back(Range::all());
	roi_rangs.push_back(Range(rang_y, rang_h + rang_y));
	roi_rangs.push_back(Range(rang_x, rang_w + rang_x));

	//crop
	Mat temp_mask_protos = maskProtos(roi_rangs).clone();
	Mat protos = temp_mask_protos.reshape(0, { seg_channels,rang_w * rang_h });
	Mat matmul_res = (maskProposals * protos).t();
	Mat masks_feature = matmul_res.reshape(1, { rang_h,rang_w });
	Mat dest, mask;

	//sigmoid
	cv::exp(-masks_feature, dest);
	dest = 1.0 / (1.0 + dest);

	int left = floor((net_width / seg_width * rang_x - params[2]) / params[0]);
	int top = floor((net_height / seg_height * rang_y - params[3]) / params[1]);
	int width = ceil(net_width / seg_width * rang_w / params[0]);
	int height = ceil(net_height / seg_height * rang_h / params[1]);

	resize(dest, mask, Size(width, height), INTER_NEAREST);
	Rect mask_rect = temp_rect - Point(left, top);
	mask_rect &= Rect(0, 0, width, height);
	mask = mask(mask_rect) > mask_threshold;
	if (mask.rows != temp_rect.height || mask.cols != temp_rect.width) { //https://github.com/UNeedCryDear/yolov8-opencv-onnxruntime-cpp/pull/30
		resize(mask, mask, temp_rect.size(), INTER_NEAREST);
	}
	output.boxMask = mask;
}

bool SegInfer(Mat& srcImg, Net& net, vector<OutputSeg>& output) {
	Mat blob;
	output.clear();
	int col = srcImg.cols;
	int row = srcImg.rows;
	Mat netInputImg;
	Vec4d params;
	LetterBox(srcImg, netInputImg, params, cv::Size(_netWidth, _netHeight));
	// imwrite("netinputimg.png", netInputImg);
	blobFromImage(netInputImg, blob, 1./255., cv::Size(_netWidth, _netHeight), cv::Scalar(), true, false);
	//**************************************************************************************************************************************************/
	//如果在其他设置没有问题的情况下但是结果偏差很大，可以尝试下用下面两句语句
	// If there is no problem with other settings, but results are a lot different from  Python-onnx , you can try to use the following two sentences
	// 
	// blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_netWidth, _netHeight), cv::Scalar(104, 117, 123), true, false);
	//$ blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_netWidth, _netHeight), cv::Scalar(114, 114,114), true, false);
	//****************************************************************************************************************************************************/
	net.setInput(blob);
	std::vector<cv::Mat> net_output_img;
	//*********************************************************************************************************************************
	//net.forward(net_output_img, net.getUnconnectedOutLayersNames());
	//opencv4.5.x和4.6.x这里输出不一致，推荐使用下面的固定名称输出
	// 如果使用net.forward(net_output_img, net.getUnconnectedOutLayersNames())，需要确认下net.getUnconnectedOutLayersNames()返回值中output0在前，output1在后，否者出错
	//
	// The outputs of opencv4.5.x and 4.6.x are inconsistent.Please make sure "output0" is in front of "output1" if you use net.forward(net_output_img, net.getUnconnectedOutLayersNames())
	//*********************************************************************************************************************************
	vector<string> output_layer_names{ "output0","output1" };
	net.forward(net_output_img, output_layer_names); //获取output的输出
    // net.forward(net_output_img, net.getUnconnectedOutLayersNames());

	std::vector<int> class_ids;//结果id数组
	std::vector<float> confidences;//结果每个id对应置信度数组
	std::vector<cv::Rect> boxes;//每个id矩形框
	std::vector<vector<float>> picked_proposals;  //output0[:,:, 5 + _className.size():net_width]===> for mask
	int net_width = net_output_img[0].size[2];
	int net_height = net_output_img[0].size[1];
	float* pdata = (float*)net_output_img[0].data;
	for (int r = 0; r < net_height; r++) {    //lines
		float box_score = pdata[4];
		if (box_score >= _classThreshold) {
			cv::Mat scores(1, _className.size(), CV_32FC1, pdata + 5);
			Point classIdPoint;
			double max_class_socre;
			minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
			max_class_socre = (float)max_class_socre;
			if (max_class_socre >= _classThreshold) {

				vector<float> temp_proto(pdata + 5 + _className.size(), pdata + net_width);
				picked_proposals.push_back(temp_proto);
				//rect [x,y,w,h]
				float x = (pdata[0] - params[2]) / params[0];  //x
				float y = (pdata[1] - params[3]) / params[1];  //y
				float w = pdata[2] / params[0];  //w
				float h = pdata[3] / params[1];  //h
				int left = MAX(int(x - 0.5 * w + 0.5), 0);
				int top = MAX(int(y - 0.5 * h + 0.5), 0);
				class_ids.push_back(classIdPoint.x);
				confidences.push_back(max_class_socre * box_score);
				boxes.push_back(Rect(left, top, int(w + 0.5), int(h + 0.5)));
			}
		}
		pdata += net_width;//下一行

	}

	//NMS
	vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidences, _classThreshold, _nmsThreshold, nms_result);
	std::cout<<"nms shape: "<<nms_result.size()<<" "<<boxes.size()<<" "<<confidences.size()<<" "<<net_width<<" "<<net_height<<"\n";
	std::vector<vector<float>> temp_mask_proposals;
	Rect holeImgRect(0, 0, srcImg.cols, srcImg.rows);
	for (int i = 0; i < nms_result.size(); ++i) {
		int idx = nms_result[i];
		OutputSeg result;
		result.id = class_ids[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx] & holeImgRect;
		temp_mask_proposals.push_back(picked_proposals[idx]);
		output.push_back(result);
	}

	MaskParams mask_params;
	mask_params.params = params;
	mask_params.srcImgShape = srcImg.size();
	mask_params.maskThreshold = _maskThreshold;
	mask_params.netHeight = _netWidth;
	mask_params.netWidth = _netWidth;
	for (int i = 0; i < temp_mask_proposals.size(); ++i) {
		GetMask2(Mat(temp_mask_proposals[i]).t(), net_output_img[1], output[i], mask_params);
	}


	//******************** ****************
	// 老版本的方案，如果上面GetMask2出错，建议使用这个。
	// If the GetMask2() still reports errors , it is recommended to use GetMask().
	//Mat mask_proposals;
	//for (int i = 0; i < temp_mask_proposals.size(); ++i) {
	//	mask_proposals.push_back(Mat(temp_mask_proposals[i]).t());
	//}
	//GetMask(mask_proposals, net_output_img[1], output, mask_params);
	//*****************************************************/


	if (output.size())
		return true;
	else
		return false;
}



///////////////////////////////////////////////////////////////////////////////
/// Show pointCloud 
///////////////////////////////////////////////////////////////////////////////

// 定义点云类型
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

// 相机内参
const double camera_factor = 1.0;
const double camera_cx = 939.2507269638458638;//563.788940;
const double camera_cy = 528.8425773400273329;//656.947067;
const double camera_fx = 1274.390077454887887;//1410.719299;
const double camera_fy = 1309.656335641865098;//1410.719299;

int depth_width = 1920;//1920;//1080;
int depth_height = 1024;//1080;//1280;
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

///////////////////////////////////////////////////////////////////////////////
/// Hand Detect 
///////////////////////////////////////////////////////////////////////////////
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
    // std::cout<<"yolo mat: "<<size.height<<" "<<size.width<<" "<<input_image.channels()<<"\n";
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
        // std::cout << "======================" <<std::endl;
        // std::cout << confidences[idx] << " " << boxes[idx].x << " " << boxes[idx].y << " " <<boxes[idx].height << " " <<boxes[idx].width << std::endl;
        // std::cout << "======================" <<std::endl;   

    }
}

///////////////////////////////////////////////////////////////////////////////
/// Depth Predict  
///////////////////////////////////////////////////////////////////////////////
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
    // std::cout<<"===> depth input shape: "<<input.size().height<<" "<<input.size().width<<" "<<input.channels()<<"\n";
    // gamma 
    double gamma_value = 1/1.8;
    std::vector<int> target_size = {1024,1920};
    //cv::imwrite("input_22.png", input);
    cv::Mat input_gamma = adjust_gamma_lut(input, gamma_value);
    //cv::imwrite("input_gamma.png", input_gamma);
    // pad
    std::vector<int> pad_size;
    std::vector<int> pdd = {1024,1920};
    if (input_gamma.size().height>1024){
        int xx = (input_gamma.size().height - 1024)/2;
        input_gamma = input_gamma(cv::Rect(0,xx,input_gamma.size().width,1024)).clone();
    }
    cv::Mat input_pad = pad_image(input_gamma, target_size, pad_size);
    //cv::imwrite("input_pad.png", input_pad);
    cv::Size size = input_pad.size();
    // std::cout<<"pad size: "<< size.height << " " << size.width << " " << input_pad.channels() << "\n";
    // model infer 
    cv::Mat blob;
    //cv::imwrite("input_blob.png", input_blob);
    cv::Mat input_blob = input_pad.clone();
    cv::dnn::blobFromImage(input_blob, blob, 1., cv::Size(target_size[1], target_size[0]), cv::Scalar(), true, false); // cv::Size(width, height) 
    // std::cout<<"=========> TEST A \n";
    // cv::dnn::blobFromImage(input_blob, blob, 1.);    
    net_depth.setInput(blob); // (b, c, h, w) 
    std::vector<cv::Mat> outputs;
    // net_depth.forward();
    auto start = std::chrono::high_resolution_clock::now();
    net_depth.forward(outputs, net_depth.getUnconnectedOutLayersNames());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    // std::cout << "===> Depth Time core: " << diff.count() << " s\n";
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
    int rows=target_size[0], cols=target_size[1];
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
    padding_box.height = target_size[0] - pad_size[1] - padding_box.y;
    // padding_box.height = 1080 - pad_size[1] - padding_box.y;
    padding_box.x = pad_size[2];
    padding_box.width = target_size[1] - pad_size[3] - padding_box.x;   
    // std::cout<<"==> result:"<<result.rows<<" "<<result.cols<<"\n"; 
    // std::cout<<"==> padding_box:"<<padding_box.x<<" "<<padding_box.y<<" "<<padding_box.width<<" "<<padding_box.height<<"\n";
    result(padding_box).copyTo(padding_result);

    // std::cout<<"===> depth output shape: "<<padding_result.size().height<<" "<<padding_result.size().width<<" "<<padding_result.channels()<<"\n";
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

    // cv::Mat frame;
    // frame = cv::imread("hand.png");
    // auto fs = frame.size();
    // std::cout<<"frame shape: "<<fs.height<<" "<<fs.width<<" "<<frame.channels()<<"\n";

    // model init 
    bool is_cuda = true;
    cv::dnn::Net net_det, det_depth, net_seg;
    load_net(net_det, is_cuda, "models/hand_det.onnx");
    load_net(net_seg, is_cuda, "models/hand_seg.onnx");
    load_net(det_depth, is_cuda, "models/hand_dep.onnx");

    // count time 
    auto start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    float fps = -1;
    int total_frames = 0;

    // pointCloud init 
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
    	// std::cout<<"frame: "<<fss.height<<" "<<fss.width<<" "<<frame.channels()<<"\n";
    	if (frame.empty()){
    	    std::cout << "frame is empty! \n";
    	}

        // detect infer 
    	auto start = std::chrono::high_resolution_clock::now();
        auto start0 = std::chrono::high_resolution_clock::now();
    	std::vector<Detection> output;
    	detect(frame, net_det, output, class_list);
    	auto end = std::chrono::high_resolution_clock::now();
    	std::chrono::duration<double> diff = end-start;
    	std::cout << "detect Time: " << diff.count() << " s\n";

    	frame_count++;
    	total_frames++;
        if(output.size()<=0) continue; 
    	// std::cout<<"outputsize: "<<detections<<"\n";
    	// std::vector<cv::Mat> boxes;
        int maxAreaIndex = 0; // 假设第一个矩形的面积最大
        int maxArea = output[0].box.area(); // 获取第一个矩形的面积
        // 遍历所有矩形，找到面积最大的矩形
        for (size_t i = 1; i < output.size(); ++i) {
            int area = output[i].box.area(); // 计算当前矩形的面积
            if (area > maxArea) {
                maxArea = area; // 更新最大面积
                maxAreaIndex = i; // 更新最大面积对应的索引
            }
        }
        auto detection = output[maxAreaIndex];
        auto box = detection.box;
        auto classId = detection.class_id;
        const auto color = colors[classId % colors.size()];
        /// TODO depth predict model input max is (896,768), need optimized
        // if(box.width>=896||box.height>=768||box.x<0||box.y<0||box.x+box.width>=frame.cols||box.y+box.height>=frame.rows){
        if(box.width>1920||box.height>1080||box.x<0||box.y<0||box.x+box.width>=frame.cols||box.y+box.height>=frame.rows){
            continue;
        }
        std::cout<<"==> detect_box:"<<box.x<<" "<<box.y<<" "<<box.width<<" "<<box.height<<"\n";

        cv::Mat box_mat = frame(box);
        // cv::imwrite("det_result.png", box_mat);

        // segment infer 
        vector<OutputSeg> seg_result;
        //生成随机颜色
        vector<Scalar> color_seg;
        srand(time(0));
        for (int i = 0; i < 80; i++) {
            int b = 255; //rand() % 256;
            int g = 255; //rand() % 256;
            int r = 255; //rand() % 256;
            color_seg.push_back(Scalar(b, g, r));
        }
        cv::Mat seg_input = box_mat.clone();
        start = std::chrono::high_resolution_clock::now();
        SegInfer(seg_input, net_seg, seg_result);
        end = std::chrono::high_resolution_clock::now();
    	diff = end-start;
    	std::cout << "seg Time: " << diff.count() << " s\n";
        // DrawPred(seg_input, seg_result, _className, color_seg);

        // depth predict 
        cv::Mat dep_input = box_mat.clone();
        start = std::chrono::high_resolution_clock::now();
        cv::Mat result = depth_predict(dep_input, det_depth);
        end = std::chrono::high_resolution_clock::now();
    	diff = end-start;
    	std::cout << "depth Time: " << diff.count() << " s\n";

        cv::Mat out11_scaled, out11_colormap;
        // result.convertTo(out11_scaled, CV_8U, 200.0, 0.0); // 将图像数据转换为8位无符号整数
        // 应用颜色映射
        // cv::applyColorMap(out11_scaled, out11_colormap, cv::COLORMAP_MAGMA);
        // cv::imwrite("depth_result0.png", out11_colormap);
        cv::Mat final;
        final.create(result.size(), result.type());
        final.setTo(cv::Scalar(0));
        for(int i=0;i<seg_result.size();i++){
            if (seg_result[i].boxMask.rows && seg_result[i].boxMask.cols > 0){
                result(seg_result[i].box).copyTo(final(seg_result[i].box));
                final(seg_result[i].box).setTo(0, seg_result[i].boxMask == 0);
                // cv::imwrite("final.png", final);
                // std::cout<<"===> seg_boxmask: "<<seg_result[i].boxMask.size().height<<" "<<seg_result[i].boxMask.size().width<<" "<<seg_result[i].boxMask.channels()<<"\n";
            }
        }
        // erode 
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
        cv::Mat final_erode;
        cv::erode(final, final_erode, kernel);
        cv::erode(final_erode, final_erode, kernel);

        // final_erode.convertTo(out11_scaled, CV_8U, 200.0, 0.0); 
        // cv::applyColorMap(out11_scaled, out11_colormap, cv::COLORMAP_MAGMA);
        // cv::imshow("Colormap Image", out11_colormap);
        // cv::waitKey(1);

        // auto end = std::chrono::high_resolution_clock::now();
    	// std::chrono::duration<double> diff = end-start;
    	// std::cout << "==> All Time: " << diff.count() << " s\n";

        // 显示结果图像
        // std::cout<<"===> depth result: "<<out11_colormap.size().height<<" "<<out11_colormap.size().width<<" "<<out11_colormap.channels()<<"\n";
        // cv::imwrite("depth_result.png", out11_colormap);
        // break; 
        // cv::imshow("Colormap Image", out11_colormap);
        // cv::waitKey(1);
        // boxes.push_back(frame(box));
        // cv::rectangle(frame, box, color, 3);
        // cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
        // cv::putText(frame, class_list[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

        // end = std::chrono::high_resolution_clock::now();
    	// diff = end-start;
    	// std::cout << "process Time: " << diff.count() << " s\n";

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
        depth2points(final_erode, cloud2);
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
    	std::cout << "cloud4 Time: " << diff.count() << " s\n" << std::endl;
        diff = end - start0;
        std::cout << "=======>>> all Time: " << diff.count() << " s\n" << std::endl;
    }
    // cv::imwrite("output-hand.png", frame);
    // for(int i=0;i<boxes.size();i++){
    //     cv::imwrite("box.png", boxes[i]);
    // }

    return 0;
}

void LetterBox(const cv::Mat& image, cv::Mat& outImage, cv::Vec4d& params, const cv::Size& newShape,
	bool autoShape, bool scaleFill, bool scaleUp, int stride, const cv::Scalar& color){
	if (false) {
		int maxLen = MAX(image.rows, image.cols);
		outImage = Mat::zeros(Size(maxLen, maxLen), CV_8UC3);
		image.copyTo(outImage(Rect(0, 0, image.cols, image.rows)));
		params[0] = 1;
		params[1] = 1;
		params[3] = 0;
		params[2] = 0;
	}

	cv::Size shape = image.size();
	float r = std::min((float)newShape.height / (float)shape.height,
		(float)newShape.width / (float)shape.width);
	if (!scaleUp)
		r = std::min(r, 1.0f);

	float ratio[2]{ r, r };
	int new_un_pad[2] = { (int)std::round((float)shape.width * r),(int)std::round((float)shape.height * r) };

	auto dw = (float)(newShape.width - new_un_pad[0]);
	auto dh = (float)(newShape.height - new_un_pad[1]);

	if (autoShape)
	{
		dw = (float)((int)dw % stride);
		dh = (float)((int)dh % stride);
	}
	else if (scaleFill)
	{
		dw = 0.0f;
		dh = 0.0f;
		new_un_pad[0] = newShape.width;
		new_un_pad[1] = newShape.height;
		ratio[0] = (float)newShape.width / (float)shape.width;
		ratio[1] = (float)newShape.height / (float)shape.height;
	}

	dw /= 2.0f;
	dh /= 2.0f;

	if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
	{
		cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
	}
	else {
		outImage = image.clone();
	}

	int top = int(std::round(dh - 0.1f));
	int bottom = int(std::round(dh + 0.1f));
	int left = int(std::round(dw - 0.1f));
	int right = int(std::round(dw + 0.1f));
	params[0] = ratio[0];
	params[1] = ratio[1];
	params[2] = left;
	params[3] = top;
	cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

void DrawPred(Mat& img, vector<OutputSeg> result, std::vector<std::string> classNames, vector<Scalar> color, bool isVideo) {
	Mat mask = img.clone();
	// imwrite("input_save.png", mask);
    cv::Mat maskt;
    maskt.create(img.size(), img.type());
    maskt.setTo(cv::Scalar(0));
	std::cout<<"result size: "<<result.size()<<"\n";
	for (int i = 0; i < result.size(); i++) {
		int left, top;
		left = result[i].box.x;
		top = result[i].box.y;
		int color_num = i;
		// imwrite("input_save.png", img);
		// std::cout<<"===> box: "<<left<<" "<<top<<" "<<result[i].box.width<<" "<<result[i].box.height<<"\n";
		rectangle(img, result[i].box, color[result[i].id], 2, 8);
		if (result[i].boxMask.rows && result[i].boxMask.cols > 0){
			mask(result[i].box).setTo(color[result[i].id], result[i].boxMask);
            maskt(result[i].box).setTo(color[result[i].id], result[i].boxMask);
		}
		string label = classNames[result[i].id] + ":" + to_string(result[i].confidence);
		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = max(top, labelSize.height);
		// rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
		putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
	}
    cv::imwrite("maskt.png", maskt);
	addWeighted(img, 0.5, mask, 0.5, 0, img); //add mask to src
	// imshow("1", img);
    std::cout<<"===> seg result: "<<img.size().height<<" "<<img.size().width<<" "<<img.channels()<<"\n";
	imwrite("seg_result_2.png", img);
	// if (!isVideo )
	// 	waitKey(); //video waiKey not in here

}