#include "QrReader.h"
#include <vector> 


#include <iostream>

using namespace cv;

QrReader::QrReader() {
}

QrReader::~QrReader() {
}

bool QrReader::find(Mat img) {
	std::cout << " test" << std::endl;
	int skipRows = 3;
	int stateCount[5] = { 0 };
	int currentState = 0;
	for (int row = skipRows - 1; row < img.rows; row += skipRows) {
		stateCount[0] = 0;
		stateCount[1] = 0;
		stateCount[2] = 0;
		stateCount[3] = 0;
		stateCount[4] = 0;
		currentState = 0;

		uchar* ptr = img.ptr<uchar>(row);
		for (int col = 0; col < img.cols; col++) {
			if (ptr[col] < 128) {
				if ((currentState % 2) == 1) {
					currentState++;
				}
				stateCount[currentState]++;
			}
			else {
				if ((currentState % 2) == 1) {
					stateCount[currentState]++;
				}
				else if (currentState != 4) {
					currentState++;
					stateCount[currentState]++;
				}
				else {
					//std::cout << " check row: " << row << " column: " << col << std::endl;
					if (checkRatio(stateCount)) {
						std::cout << " top row: " << row << " column: " << col << std::endl;
					}
					else {
						currentState = 3;
						stateCount[0] = stateCount[2];
						stateCount[1] = stateCount[3];
						stateCount[2] = stateCount[4];
						stateCount[3] = 1;
						stateCount[4] = 0;
						continue;
					}
					currentState = 0;
					stateCount[0] = 0;
					stateCount[1] = 0;
					stateCount[2] = 0;
					stateCount[3] = 0;
					stateCount[4] = 0;
				}
			}
		}
	}
	return false;
}

bool QrReader::checkRatio(int stateCount[]) {
	int totalFinderSize = 0;
	for (int i = 0; i<5; i++) { 
		int count = stateCount[i]; 
		totalFinderSize += count; 
		if (count == 0) 
			return false;
	}
	if (totalFinderSize<7) 
		return false; // Calculate the size of one module
 	int moduleSize = static_cast<int>(ceil(totalFinderSize / 7.0)); 
	int maxVariance = moduleSize/2; 
	bool retVal= ((abs(moduleSize - (stateCount[0])) < maxVariance) && 
		(abs(moduleSize - (stateCount[1])) < maxVariance) && 
		(abs(3*moduleSize - (stateCount[2])) < 3*maxVariance) && 
		(abs(moduleSize - (stateCount[3])) < maxVariance) && 
		(abs(moduleSize - (stateCount[4])) < maxVariance)); 
	return retVal; 
}