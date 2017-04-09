#pragma once
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 
class QrReader
{
public:
	QrReader(cv::Mat&);
	virtual ~QrReader();
	bool QrReader::find();
private:
	cv::Mat img;

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

		retVal &= (5 * abs(stateCount[0] + stateCount[1] + stateCount[2] + stateCount[3] + stateCount[4] - originalStateCountTotal) <= originalStateCountTotal);
		retVal &= checkRatio(stateCount);

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

	bool handlePossibleCenter(float & row, float & col, int stateCountTotal);
};

