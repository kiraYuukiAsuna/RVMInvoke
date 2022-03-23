#include "RVMInvoke/RVMInvoke.h"

int main()
{
	std::string onnx_path = "rvm_mobilenetv3_fp32.onnx";
	std::string video_path = "TEST_01.mp4";
	std::string output_path = "A_TEST_01.mp4";

	RobustVideoMatting rvm(onnx_path, 12); // 12 threads
	std::vector<MattingContent> contents;

	rvm.detect_video(video_path, output_path, contents, cv::Size(960, 540), 0.25, cv::Scalar(0,0,0), false, 0);

	//in
	// onnx path -> dev mode only
	// thread num -> dev mode only

	// in
	// video path -> ok
	// output path -> ok
	// content -> dev mode only
	// save content -> dev mode only
	// downsample ratio -> ok
	// write fps -> ok
	// input net img size -> ok
	// backfround color -> ok

	//in
	// img mat -> ok
	// content -> dev mode only
	// downsample ratio -> ok
	// video mode -> dev mode only
	// input net img size -> ok
	// backfround color -> ok

	return 0;
}
