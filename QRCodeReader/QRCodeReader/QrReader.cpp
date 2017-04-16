#include "QrReader.h"
#include <vector> 


#include <iostream>

using namespace cv;

QrReader::QrReader(cv::Mat& imgp) 
	: img(imgp) {
	cvtColor(img, img, CV_BGR2GRAY);
	threshold(img, img, 128, 255, THRESH_BINARY);
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
						
						int stateCountTotal = stateCount[0] + stateCount[1] + stateCount[2] + stateCount[3] + stateCount[4];
						float exactRow = static_cast<float>(row);
						float exactCol = static_cast<float>(col - (stateCount[4] + stateCount[3] + stateCount[2] / 2));

						if (checkPossibleCenter(exactRow, exactCol, stateCountTotal))
						{	
							if (!checkCenterDuplicate(exactRow, exactCol, stateCountTotal))
							{
								std::cout << "exact row " << exactRow << " col " << exactCol << std::endl;
							}
						}

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
	return selectBestPatterns();
}

bool QrReader::checkRatio(int stateCount[]) {
	int totalFinderSize = 0;
	for (int i = 0; i<5; i++) {
		totalFinderSize += stateCount[i];
		if (stateCount[i] == 0)
			return false;
	}
	if (totalFinderSize<7) 
		return false;


 	int moduleSize = static_cast<int>(ceil(totalFinderSize / 7.0)); 
	int maxVariance = moduleSize/2; 
	bool retVal= ((abs(moduleSize - (stateCount[0])) < maxVariance) && 
		(abs(moduleSize - (stateCount[1])) < maxVariance) && 
		(abs(3*moduleSize - (stateCount[2])) < 3*maxVariance) && 
		(abs(moduleSize - (stateCount[3])) < maxVariance) && 
		(abs(moduleSize - (stateCount[4])) < maxVariance)); 
	return retVal; 
}

bool QrReader::checkPossibleCenter(float & row, float & col, int stateCountTotal)
{
	//std::cout << " row: " << row << " col: " << col << " bool: " << retval << std::endl;

	//std::cout << " row: " << row << " col: " << col << std::endl;
	//bool retVal = crossCheck<1, 0>(row, col, stateCountTotal);

	//std::cout << " row: " << row << " col: " << col << " bool: " << retVal << std::endl;

	//retVal = retVal && crossCheck<0, 1>(row, col, stateCountTotal);
	//std::cout << " row: " << row << " col: " << col << " bool: " << retVal << std::endl;
	//float rowDummy = row;
	//float colDummy = col;
	//retVal = retVal && crossCheck<1, 1>(rowDummy, colDummy, stateCountTotal);
	//std::cout << " row: " << row << " col: " << col << " bool: " << retVal << std::endl;

	//return retVal;
	return crossCheck<1, 0>(row, col, stateCountTotal) &&
		crossCheck<0, 1>(row, col, stateCountTotal) &&
		crossCheck<1, 1>(float(row), float(col), stateCountTotal);
}

bool QrReader::checkCenterDuplicate(float row, float col, int stateCountTotal)
{
	float estimatedModuleSize = stateCountTotal / 7.0f;
	int index;
	for (index = 0; index < possibleCenters.size() && !possibleCenters[index]->aboutEquals(row, col, estimatedModuleSize); index++);
	bool existing = index < possibleCenters.size();
	if (existing)
		possibleCenters[index]->combineEstimate(row, col, estimatedModuleSize);
	else
		possibleCenters.push_back(new FinderPattern(row, col, estimatedModuleSize));

	return existing;
}

bool QrReader::selectBestPatterns()
{
	if (possibleCenters.size() > 3)
	{
		float totalModuleSize = 0.0f;
		float square = 0.0f;
		for(std::vector<FinderPattern*>::iterator it = possibleCenters.begin(); it != possibleCenters.end(); it++)
		{
			float size = (*it)->getModuleSize();
			totalModuleSize += size;
			square += size * size;
		}

		float average = totalModuleSize / possibleCenters.size();
		float stdDev = cv::sqrt(square / possibleCenters.size() - average * average);

		std::sort(possibleCenters.begin(), possibleCenters.end(), FurthestFromAverageComparator(average));

		float limit = cv::max(0.2f * average, stdDev);

		for (int i = 0; i < possibleCenters.size() && possibleCenters.size() > 3; i++)
		{
			if (cv::abs(possibleCenters[i]->getModuleSize() - average) > limit)
			{
				possibleCenters.erase(possibleCenters.begin() + i);
				i--;
			}
		}
	}

	if (possibleCenters.size() > 3)
	{
		float totalModuleSize = 0.0f;
		for (std::vector<FinderPattern*>::iterator it = possibleCenters.begin(); it != possibleCenters.end(); it++)
		{
			totalModuleSize += (*it)->getModuleSize();
		}

		float average = totalModuleSize / possibleCenters.size();

		std::sort(possibleCenters.begin(), possibleCenters.end(), CenterComparator(average));

		possibleCenters.erase(possibleCenters.begin() + 3, possibleCenters.end());
	}

	if (3 == possibleCenters.size())
	{
		float zeroOneDistance = distance(*possibleCenters[0], *possibleCenters[1]);
		float oneTwoDistance = distance(*possibleCenters[1], *possibleCenters[2]);
		float zeroTwoDistance = distance(*possibleCenters[0], *possibleCenters[2]);

		FinderPattern *pointA, *pointB, *pointC;

		if (oneTwoDistance >= zeroOneDistance && oneTwoDistance >= zeroTwoDistance)
		{
			std::cout << "12" << std::endl;
			pointB = possibleCenters[0];
			pointA = possibleCenters[1];
			pointC = possibleCenters[2];
		}
		else if (zeroTwoDistance >= oneTwoDistance && zeroTwoDistance >= zeroOneDistance)
		{
			std::cout << "02" << std::endl;
			pointB = possibleCenters[1];
			pointA = possibleCenters[0];
			pointC = possibleCenters[2];
		}
		else
		{
			std::cout << "01" << std::endl;
			pointB = possibleCenters[2];
			pointA = possibleCenters[0];
			pointC = possibleCenters[1];
		}

		if (crossProductZ(*pointA, *pointB, *pointC) < 0.0f)
		{
			cv::swap(pointA, pointC);
		}

		possibleCenters[0] = pointA;
		possibleCenters[1] = pointB;
		possibleCenters[2] = pointC;


		std::cout << "row " << possibleCenters[0]->getRow() << " col " << possibleCenters[0]->getCol() << std::endl;
		std::cout << "row " << possibleCenters[1]->getRow() << " col " << possibleCenters[1]->getCol() << std::endl;
		std::cout << "row " << possibleCenters[2]->getRow() << " col " << possibleCenters[2]->getCol() << std::endl;
	}
	return 3 == possibleCenters.size();
}
//
//void QrReader::processFinderPatternInfo()
//{
//	FinderPattern *topLeft = possibleCenters[1];
//	FinderPattern *topRight = possibleCenters[2];
//	FinderPattern *bottomLeft = possibleCenters[0];
//
//	float moduleSize = calculateModuleSize(topLeft, topRight, bottomLeft);
//	if (moduleSize < 1.0f)
//	{
//		return null;
//	}
//	int dimension;
//	if (!computeDimension(topLeft, topRight, bottomLeft, moduleSize, out dimension))
//		return null;
//	Internal.Version provisionalVersion = Internal.Version.getProvisionalVersionForDimension(dimension);
//	if (provisionalVersion == null)
//		return null;
//	int modulesBetweenFPCenters = provisionalVersion.DimensionForVersion - 7;
//
//	AlignmentPattern alignmentPattern = null;
//	// Anything above version 1 has an alignment pattern
//	if (provisionalVersion.AlignmentPatternCenters.Length > 0)
//	{
//
//		// Guess where a "bottom right" finder pattern would have been
//		float bottomRightX = topRight.X - topLeft.X + bottomLeft.X;
//		float bottomRightY = topRight.Y - topLeft.Y + bottomLeft.Y;
//
//		// Estimate that alignment pattern is closer by 3 modules
//		// from "bottom right" to known top left location
//		//UPGRADE_WARNING: Data types in Visual C# might be different.  Verify the accuracy of narrowing conversions. "ms-help://MS.VSCC.v80/dv_commoner/local/redirect.htm?index='!DefaultContextWindowIndex'&keyword='jlca1042'"
//		float correctionToTopLeft = 1.0f - 3.0f / (float)modulesBetweenFPCenters;
//		//UPGRADE_WARNING: Data types in Visual C# might be different.  Verify the accuracy of narrowing conversions. "ms-help://MS.VSCC.v80/dv_commoner/local/redirect.htm?index='!DefaultContextWindowIndex'&keyword='jlca1042'"
//		int estAlignmentX = (int)(topLeft.X + correctionToTopLeft * (bottomRightX - topLeft.X));
//		//UPGRADE_WARNING: Data types in Visual C# might be different.  Verify the accuracy of narrowing conversions. "ms-help://MS.VSCC.v80/dv_commoner/local/redirect.htm?index='!DefaultContextWindowIndex'&keyword='jlca1042'"
//		int estAlignmentY = (int)(topLeft.Y + correctionToTopLeft * (bottomRightY - topLeft.Y));
//
//		// Kind of arbitrary -- expand search radius before giving up
//		for (int i = 4; i <= 16; i <<= 1)
//		{
//			alignmentPattern = findAlignmentInRegion(moduleSize, estAlignmentX, estAlignmentY, (float)i);
//			if (alignmentPattern == null)
//				continue;
//			break;
//		}
//		// If we didn't find alignment pattern... well try anyway without it
//	}
//
//	PerspectiveTransform transform = createTransform(topLeft, topRight, bottomLeft, alignmentPattern, dimension);
//
//	BitMatrix bits = sampleGrid(image, transform, dimension);
//	if (bits == null)
//		return null;
//
//	ResultPoint[] points;
//	if (alignmentPattern == null)
//	{
//		points = new ResultPoint[]{ bottomLeft, topLeft, topRight };
//	}
//	else
//	{
//		points = new ResultPoint[]{ bottomLeft, topLeft, topRight, alignmentPattern };
//	}
//	return new DetectorResult(bits, points);
//}
