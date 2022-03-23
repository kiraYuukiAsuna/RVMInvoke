#include "RVMInvoke/RVMInvoke.h"

int main()
{
	std::string onnx_path = "rvm_mobilenetv3_fp32.onnx";
	std::string video_path = "TEST_20.mp4";
	std::string output_path = "A_TEST_20.mp4";

	auto* rvm = new RobustVideoMatting(onnx_path, 12); // 12 threads
	std::vector<MattingContent> contents;

	// 1. video matting.
	rvm->detect_video(video_path, output_path, contents, false, 0.25);

	delete rvm;
	return 0;
}
