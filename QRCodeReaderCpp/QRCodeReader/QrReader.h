#pragma once
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 
#include <vector>
#include "FinderPattern.h"
class QrReader
{
public:
	QrReader(cv::Mat&);
	virtual ~QrReader() {}
	bool QrReader::find();
private:
	cv::Mat img;
	std::vector<FinderPattern*> possibleCenters;

	bool QrReader::checkRatio(int []);

	template<class T, int rowOffset, int colOffset> inline bool getMovedPtr(T* &ptr, int row, int col, int offset)
	{
		row += rowOffset * offset;
		col += colOffset * offset;

		bool retVal = 0 < row && img.rows > row && 0 < col && img.cols > col;
		if (retVal)
			ptr = img.ptr<T>(row, col);

		return retVal;
	}

	template<int rowOffset, int colOffset> inline bool movedIsWhite(bool &value, int row, int col, int offset)
	{
		uchar * ptr;
		bool retVal = getMovedPtr<uchar, rowOffset, colOffset>(ptr, row, col, offset);
		if (retVal)
			value = 0 != *ptr;
		return retVal;
	}

	template<int rowOffset, int colOffset> bool crossCheck(float & row, float & col, int originalStateCountTotal)
	{
		int iRow = static_cast<int>(row);
		int iCol = static_cast<int>(col);
		int stateCount[5] = { 0 };
		bool isWhite;
		int offset = 0;
		bool retVal = movedIsWhite<rowOffset, colOffset>(isWhite, iRow, iCol, offset);
		
		for (int state = 2; state >= 0; state--)
		{
			while (retVal && (isWhite == (state % 2 != 0)))
			{
				stateCount[state]++;
				offset--;
				retVal = stateCount[state] <= originalStateCountTotal && movedIsWhite<rowOffset, colOffset>(isWhite, iRow, iCol, offset);
			}
		}
		
		offset = 1;
		retVal = movedIsWhite<rowOffset, colOffset>(isWhite, iRow, iCol, offset);

		for (int state = 2; state <= 4; state++)
		{
			while (retVal && (isWhite == (state % 2 != 0)))
			{
				stateCount[state]++;
				offset++;
				retVal = stateCount[state] <= originalStateCountTotal && movedIsWhite<rowOffset, colOffset>(isWhite, iRow, iCol, offset);
			}
		}
		
		int multiplier = static_cast<int>(5 / sqrt(rowOffset * rowOffset + colOffset * colOffset));

		//if (rowOffset || colOffset) std::cout << retVal << " :" << (stateCount[0] + stateCount[1] + stateCount[2] + stateCount[3] + stateCount[4]) << " :" << originalStateCountTotal;
		retVal = retVal && (multiplier * abs(stateCount[0] + stateCount[1] + stateCount[2] + stateCount[3] + stateCount[4] - originalStateCountTotal) <= originalStateCountTotal);
		//if (rowOffset || colOffset) std::cout << " " << retVal;
		retVal = retVal && checkRatio(stateCount);
		//if (rowOffset || colOffset) std::cout << " " << retVal << std::endl;
		if (retVal)
		{
			float centerDelta = (offset - (stateCount[4] + stateCount[3] + stateCount[2] / 2.0f));
			
			if(0 != rowOffset)
				row = iRow + centerDelta * rowOffset;
			if(0 != colOffset)
				col = iCol + centerDelta * colOffset;
		}
		return retVal;
	}

	bool checkPossibleCenter(float & row, float & col, int stateCountTotal);

	bool checkCenterDuplicate(float row, float col, int stateCountTotal);

	bool selectBestPatterns();

	float distance(const FinderPattern & lhs, const FinderPattern & rhs)
	{
		return cv::sqrt((lhs.getRow() - rhs.getRow()) * (lhs.getRow() - rhs.getRow()) + 
			(lhs.getCol() - rhs.getCol()) * (lhs.getCol() - rhs.getCol()));
	}

	float crossProductZ(const FinderPattern & pointA, const FinderPattern & pointB, const FinderPattern & pointC)
	{
		return ((pointC.getCol() - pointB.getCol()) * (pointA.getRow() - pointB.getRow())) - 
			((pointC.getRow() - pointB.getRow()) * (pointA.getCol() - pointB.getCol()));
	}

	class FurthestFromAverageComparator
	{
	private:
		float average;
	public:
		FurthestFromAverageComparator(float f) { average = f; }

		bool operator() (FinderPattern *lhs, FinderPattern *rhs)
		{ return cv::abs(lhs->getModuleSize() - average) > cv::abs(rhs->getModuleSize() - average); }
	};

	class CenterComparator

	{
	private:
		float average;
	public:
		CenterComparator(float f) { average = f; }

		bool operator() (FinderPattern *lhs, FinderPattern *rhs)
		{
			return (lhs->getCount() == rhs->getCount()) ?
				cv::abs(lhs->getModuleSize() - average) > cv::abs(rhs->getModuleSize() - average) : 
				lhs->getCount() < rhs->getCount();
		}
	};

	//bool computeDimension(ResultPoint topLeft, ResultPoint topRight, ResultPoint bottomLeft, float moduleSize, out int dimension)
	//{
	//	int tltrCentersDimension = MathUtils.round(ResultPoint.distance(topLeft, topRight) / moduleSize);
	//	int tlblCentersDimension = MathUtils.round(ResultPoint.distance(topLeft, bottomLeft) / moduleSize);
	//	dimension = ((tltrCentersDimension + tlblCentersDimension) >> 1) + 7;
	//	switch (dimension & 0x03)
	//	{
	//		// mod 4
	//	case 0:
	//		dimension++;
	//		break;
	//		// 1? do nothing
	//	case 2:
	//		dimension--;
	//		break;
	//	case 3:
	//		return true;
	//	}
	//	return true;
	//}


	//float sizeOfBlackWhiteBlackRunBothWays(int fromX, int fromY, int toX, int toY)
	//{

	//	float result = sizeOfBlackWhiteBlackRun(fromX, fromY, toX, toY);

	//	// Now count other way -- don't run off image though of course
	//	float scale = 1.0f;
	//	int otherToX = fromX - (toX - fromX);
	//	if (otherToX < 0)
	//	{
	//		//UPGRADE_WARNING: Data types in Visual C# might be different.  Verify the accuracy of narrowing conversions. "ms-help://MS.VSCC.v80/dv_commoner/local/redirect.htm?index='!DefaultContextWindowIndex'&keyword='jlca1042'"
	//		scale = (float)fromX / (float)(fromX - otherToX);
	//		otherToX = 0;
	//	}
	//	else if (otherToX >= image.Width)
	//	{
	//		//UPGRADE_WARNING: Data types in Visual C# might be different.  Verify the accuracy of narrowing conversions. "ms-help://MS.VSCC.v80/dv_commoner/local/redirect.htm?index='!DefaultContextWindowIndex'&keyword='jlca1042'"
	//		scale = (float)(image.Width - 1 - fromX) / (float)(otherToX - fromX);
	//		otherToX = image.Width - 1;
	//	}
	//	//UPGRADE_WARNING: Data types in Visual C# might be different.  Verify the accuracy of narrowing conversions. "ms-help://MS.VSCC.v80/dv_commoner/local/redirect.htm?index='!DefaultContextWindowIndex'&keyword='jlca1042'"
	//	int otherToY = (int)(fromY - (toY - fromY) * scale);

	//	scale = 1.0f;
	//	if (otherToY < 0)
	//	{
	//		//UPGRADE_WARNING: Data types in Visual C# might be different.  Verify the accuracy of narrowing conversions. "ms-help://MS.VSCC.v80/dv_commoner/local/redirect.htm?index='!DefaultContextWindowIndex'&keyword='jlca1042'"
	//		scale = (float)fromY / (float)(fromY - otherToY);
	//		otherToY = 0;
	//	}
	//	else if (otherToY >= image.Height)
	//	{
	//		//UPGRADE_WARNING: Data types in Visual C# might be different.  Verify the accuracy of narrowing conversions. "ms-help://MS.VSCC.v80/dv_commoner/local/redirect.htm?index='!DefaultContextWindowIndex'&keyword='jlca1042'"
	//		scale = (float)(image.Height - 1 - fromY) / (float)(otherToY - fromY);
	//		otherToY = image.Height - 1;
	//	}
	//	//UPGRADE_WARNING: Data types in Visual C# might be different.  Verify the accuracy of narrowing conversions. "ms-help://MS.VSCC.v80/dv_commoner/local/redirect.htm?index='!DefaultContextWindowIndex'&keyword='jlca1042'"
	//	otherToX = (int)(fromX + (otherToX - fromX) * scale);

	//	result += sizeOfBlackWhiteBlackRun(fromX, fromY, otherToX, otherToY);
	//	return result - 1.0f; // -1 because we counted the middle pixel twice
	//}
	//float sizeOfBlackWhiteBlackRun(int fromX, int fromY, int toX, int toY)
	//{
	//	// Mild variant of Bresenham's algorithm;
	//	// see http://en.wikipedia.org/wiki/Bresenham's_line_algorithm
	//	bool steep = Math.Abs(toY - fromY) > Math.Abs(toX - fromX);
	//	if (steep)
	//	{
	//		int temp = fromX;
	//		fromX = fromY;
	//		fromY = temp;
	//		temp = toX;
	//		toX = toY;
	//		toY = temp;
	//	}

	//	int dx = Math.Abs(toX - fromX);
	//	int dy = Math.Abs(toY - fromY);
	//	int error = -dx >> 1;
	//	int xstep = fromX < toX ? 1 : -1;
	//	int ystep = fromY < toY ? 1 : -1;

	//	// In black pixels, looking for white, first or second time.
	//	int state = 0;
	//	// Loop up until x == toX, but not beyond
	//	int xLimit = toX + xstep;
	//	for (int x = fromX, y = fromY; x != xLimit; x += xstep)
	//	{
	//		int realX = steep ? y : x;
	//		int realY = steep ? x : y;

	//		// Does current pixel mean we have moved white to black or vice versa?
	//		// Scanning black in state 0,2 and white in state 1, so if we find the wrong
	//		// color, advance to next state or end if we are in state 2 already
	//		if ((state == 1) == image[realX, realY])
	//		{
	//			if (state == 2)
	//			{
	//				return MathUtils.distance(x, y, fromX, fromY);
	//			}
	//			state++;
	//		}
	//		error += dy;
	//		if (error > 0)
	//		{
	//			if (y == toY)
	//			{


	//				break;
	//			}
	//			y += ystep;
	//			error -= dx;
	//		}
	//	}
	//	// Found black-white-black; give the benefit of the doubt that the next pixel outside the image
	//	// is "white" so this last point at (toX+xStep,toY) is the right ending. This is really a
	//	// small approximation; (toX+xStep,toY+yStep) might be really correct. Ignore this.
	//	if (state == 2)
	//	{
	//		return MathUtils.distance(toX + xstep, toY, fromX, fromY);
	//	}
	//	// else we didn't find even black-white-black; no estimate is really possible
	//	return Single.NaN;

	//}
	//float calculateModuleSizeOneWay(FinderPattern pattern, FinderPattern otherPattern)
	//{
	//	//UPGRADE_WARNING: Data types in Visual C# might be different.  Verify the accuracy of narrowing conversions. "ms-help://MS.VSCC.v80/dv_commoner/local/redirect.htm?index='!DefaultContextWindowIndex'&keyword='jlca1042'"
	//	float moduleSizeEst1 = sizeOfBlackWhiteBlackRunBothWays((int)pattern.X, (int)pattern.Y, (int)otherPattern.X, (int)otherPattern.Y);
	//	//UPGRADE_WARNING: Data types in Visual C# might be different.  Verify the accuracy of narrowing conversions. "ms-help://MS.VSCC.v80/dv_commoner/local/redirect.htm?index='!DefaultContextWindowIndex'&keyword='jlca1042'"
	//	float moduleSizeEst2 = sizeOfBlackWhiteBlackRunBothWays((int)otherPattern.X, (int)otherPattern.Y, (int)pattern.X, (int)pattern.Y);
	//	if (Single.IsNaN(moduleSizeEst1))
	//	{
	//		return moduleSizeEst2 / 7.0f;
	//	}
	//	if (Single.IsNaN(moduleSizeEst2))
	//	{
	//		return moduleSizeEst1 / 7.0f;
	//	}
	//	// Average them, and divide by 7 since we've counted the width of 3 black modules,
	//	// and 1 white and 1 black module on either side. Ergo, divide sum by 14.
	//	return (moduleSizeEst1 + moduleSizeEst2) / 14.0f;
	//}
	//float calculateModuleSize(FinderPattern topLeft, FinderPattern topRight, FinderPattern bottomLeft)
	//{
	//	// Take the average
	//	return (calculateModuleSizeOneWay(topLeft, topRight) + calculateModuleSizeOneWay(topLeft, bottomLeft)) / 2.0f;
	//}

	void processFinderPatternInfo();
};

