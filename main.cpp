#include "RVMInvoke/RVMInvoke.h"

int main()
{
	std::string onnx_path = "rvm_mobilenetv3_fp32.onnx";
	std::string video_path = "TEST_01.mp4";
	std::string output_path = "A_TEST_01.mp4";

	auto* rvm = new RobustVideoMatting(onnx_path, 12); // 12 threads
	std::vector<MattingContent> contents;

	rvm->detect_video(video_path, output_path, contents, cv::Size(960, 540), false, 0.25, 0);

	delete rvm;

	//in
	// onnx path -> dev mode only
	// thread num -> dev mode only

	// in
	// video path -> ok
	// output path -> ok
	// content -> dev mode only
	// save content -> dev mode only
	// downsample ratio -> user mode
	// write fps -> user mode
	// input net img size -> user mode

	//in
	// img mat -> ok
	// content -> dev mode only
	// downsample ratio -> user mode
	// video mode -> user mode
	// input net img size -> user mode

	return 0;
}
