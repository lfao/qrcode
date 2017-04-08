#pragma once
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 
class QrReader
{
public:
	QrReader();
	virtual ~QrReader();
	bool QrReader::find(cv::Mat);
private:
	bool QrReader::checkRatio(int []);
};

