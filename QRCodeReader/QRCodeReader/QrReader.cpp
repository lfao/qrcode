#include "QrReader.h"
#include <vector> 


#include <iostream>

using namespace cv;

QrReader::QrReader(cv::Mat& imgp) 
	: img(imgp) {
	cvtColor(img, img, CV_BGR2GRAY);
	threshold(img, img, 128, 255, THRESH_BINARY);
}

QrReader::~QrReader() {
}

bool QrReader::find() {
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
			if (ptr[col] == 0) {
				if ((currentState % 2) == 1) {
					currentState++;
				}
				stateCount[currentState]++;
			} else {
				if ((currentState % 2) == 1) {
					stateCount[currentState]++;
				} else if (currentState != 4) {
					currentState++;
					stateCount[currentState]++;
				} else {
					if (checkRatio(stateCount)) {
						//std::cout << " top row: " << row << " column: " << col << std::endl;
						
						float exactRow = static_cast<float>(row);
						float exactCol = static_cast<float>(col - (stateCount[4] + stateCount[3] + stateCount[2] / 2));

						if (handlePossibleCenter(exactRow, exactCol,
							stateCount[0] + stateCount[1] + stateCount[2] + stateCount[3] + stateCount[4]))
							std::cout << "exact row " << exactRow << " col " << exactCol << std::endl;


						currentState = 0;
						stateCount[0] = 0;
						stateCount[1] = 0;
						stateCount[2] = 0;
						stateCount[3] = 0;
						stateCount[4] = 0;
					} else {
						currentState = 3;
						stateCount[0] = stateCount[2];
						stateCount[1] = stateCount[3];
						stateCount[2] = stateCount[4];
						stateCount[3] = 1;
						stateCount[4] = 0;
					}
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


bool QrReader::handlePossibleCenter(float & row, float & col, int stateCountTotal)
{
	//std::cout << " row: " << row << " col: " << col << std::endl;
	//bool retval = crossCheck<0, 1>(row, col, stateCountTotal);

	//std::cout << " row: " << row << " col: " << col << " bool: " << retval << std::endl;

	//std::cout << " row: " << row << " col: " << col << std::endl;
	bool retval = crossCheck<1, 0>(row, col, stateCountTotal);

	//std::cout << " row: " << row << " col: " << col << " bool: " << retval << std::endl;

	retval &= crossCheck<0, 1>(row, col, stateCountTotal);
	//std::cout << " row: " << row << " col: " << col << " bool: " << retval << std::endl;
	float rowDummy = row;
	float colDummy = col;
	retval &= crossCheck<1, 1>(rowDummy, colDummy, stateCountTotal);
	//std::cout << " row: " << row << " col: " << col << " bool: " << retval << std::endl;

	return retval;
	//return crossCheck<1, 0>(row, col, stateCountTotal) &&
	//	crossCheck<0, 1>(row, col, stateCountTotal) &&
	//	crossCheck<1, 1>(float(row), float(col), stateCountTotal);
}
	//{
	//	std::cout << "valid" << std::endl;
	//	float estimatedModuleSize = stateCountTotal / 7.0f;
	//	bool found = false;

	//	//for (int index = 0; index < possibleCenters.Count; index++)
	//	//{
	//	//	var center = possibleCenters[index];
	//	//	// Look for about the same center and module size:
	//	//	if (center.aboutEquals(estimatedModuleSize, centerI.Value, centerJ.Value))
	//	//	{
	//	//		possibleCenters.RemoveAt(index);
	//	//		possibleCenters.Insert(index, center.combineEstimate(centerI.Value, centerJ.Value, estimatedModuleSize));

	//	//		found = true;
	//	//		break;
	//	//	}
	//	//}
	//	//if (!found)
	//	//{
	//	//	var point = new FinderPattern(centerJ.Value, centerI.Value, estimatedModuleSize);

	//	//	possibleCenters.Add(point);
	//	//	if (resultPointCallback != null)
	//	//	{

	//	//		resultPointCallback(point);
	//	//	}
	//	//}
	//	return true;
	//}
	//return false;


