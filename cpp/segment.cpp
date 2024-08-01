#include <fstream>
#include <vector>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace cv::dnn;

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

std::vector<std::string> load_class_list(){
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
	imwrite("netinputimg.png", netInputImg);
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


int main(int argc, char **argv){

    bool is_cuda = true;
    cv::dnn::Net net_seg;
	std::cout<<"===> model init start\n";
    load_net(net_seg, is_cuda, "config_files/hand_seg.onnx");
	std::cout<<"===> model init done\n";

    // count time 
    auto start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    float fps = -1;
    int total_frames = 0;

    while(true){
    	cv::Mat input = cv::imread("box_mat.png");
    	cv::Mat frame = input.clone();
    	auto fss = input.size();

    	// cv::repeat(input, 3, 2, frame); // (1080,1920) -> (1080,1920,3)
    	fss = frame.size();
    	if (frame.empty()){
    	    std::cout << "frame is empty! \n";
    	}

    	auto start = std::chrono::high_resolution_clock::now();
    	
		vector<OutputSeg> output;
		//生成随机颜色
		vector<Scalar> color;
		srand(time(0));
		for (int i = 0; i < 80; i++) {
			int b = rand() % 256;
			int g = rand() % 256;
			int r = rand() % 256;
			color.push_back(Scalar(b, g, r));
		}
		// imwrite("input_frame.png", frame);
		SegInfer(frame, net_seg, output);
		DrawPred(frame, output, _className, color);

    	auto end = std::chrono::high_resolution_clock::now();
    	std::chrono::duration<double> diff = end-start;
    	std::cout << "===> Model ALL RUN Time: " << diff.count() << " s\n";

		break;
    }

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
	std::cout<<"result size: "<<result.size()<<"\n";
	for (int i = 0; i < result.size(); i++) {
		int left, top;
		left = result[i].box.x;
		top = result[i].box.y;
		int color_num = i;
		// imwrite("input_save.png", img);
		std::cout<<"===> box: "<<left<<" "<<top<<" "<<result[i].box.width<<" "<<result[i].box.height<<"\n";
		rectangle(img, result[i].box, color[result[i].id], 2, 8);
		if (result[i].boxMask.rows && result[i].boxMask.cols > 0){
			mask(result[i].box).setTo(color[result[i].id], result[i].boxMask);
		}
		string label = classNames[result[i].id] + ":" + to_string(result[i].confidence);
		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = max(top, labelSize.height);
		//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
		putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
	}
	addWeighted(img, 0.5, mask, 0.5, 0, img); //add mask to src
	// imshow("1", img);
	imwrite("seg_result.png", img);
	if (!isVideo )
		waitKey(); //video waiKey not in here

}