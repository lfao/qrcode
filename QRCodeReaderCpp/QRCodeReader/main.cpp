#include <iostream> 
#include <vector> 
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 
#include "QrReader.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[], char *envp[]) {
	////cv::VideoCapture capture = VideoCapture(1);
	//Mat image = imread("C:\\Users\\leonh\\Source\\Repos\\qrcode\\QRCodeReader\\5.png");
	//Mat image = imread("C:\\Users\\leonh\\OneDrive\\Bildung&Beruf\\2017 Erasmus Comillas\\Computer Vision\\project\\qrcode - datasets\\datasets\\lighting\\20110817_013.jpg");
	//Mat image = imread("C:\\Users\\leonh\\OneDrive\\Bildung&Beruf\\2017 Erasmus Comillas\\Computer Vision\\project\\qrcode - datasets\\datasets\\lighting\\20110817_013.jpg");
	//Mat image = imread("20110817_013.jpg");
	//Mat image = imread("IMG_20120226_165408.jpg");
	//Mat image = imread("IMG_2743.JPG");
	Mat image = imread("IMG_2713.JPG");
	//Mat image = imread("");
	//Mat image = imread("");
	//Mat image = imread("");
	//Mat image = imread("");
	//Mat image = imread("");
	//Mat image = imread("");
	if (image.empty())
		std::cout << "failed to open img.jpg" << std::endl;
	
	//Mat imgBw = image;

	//imshow("threshold", imgBw);
	//
	QrReader theQrReader = QrReader(image);
	std::cout << theQrReader.find();
	//imshow("image", image);


	cout << "Finish" << endl;
	////char a;
	////cin >> a;

	//cv::VideoCapture capture = VideoCapture(1);
	//QrReader qr = QrReader();

	//if (!capture.isOpened())
	//	printf("Unable to open camera");
	//Mat image;
	//Mat imgBW;
	//while (true)
	//{
	//	capture >> image;
	//	cvtColor(image, imgBW, CV_BGR2GRAY);
	//	threshold(imgBW, imgBW, 128, 255, THRESH_BINARY);
	//	imshow("image", imgBW);
	//	//bool found = qr.find(imgBW);
	///*	if (found)
	//		qr.drawFinders(imgBW);
	//	imshow("image", imgBW);
	//	waitKey(30);*/
	//}
	//waitKey(0);


	//VideoCapture stream1(0); //0 is the id of video device.0 if you have only one camera.

	//if (!stream1.isOpened()) { //check if video device has been initialised
	//	cout << "cannot open camera";
	//}

	//QrReader theQrReader = QrReader();
	////unconditional loop
	while (true) {
	//	Mat cameraFrame, imgBw;
	//	stream1.read(cameraFrame);
	//	imshow("cam", cameraFrame);
	//	cvtColor(cameraFrame, imgBw, CV_BGR2GRAY);
	//	theQrReader.find(imgBw);
		if ((char)waitKey(30) >= 0)
		{
			break;
		}
	}
	//return 0;
}