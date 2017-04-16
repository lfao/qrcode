#include "FinderPattern.h"
#include <opencv2/imgproc/imgproc.hpp> 
bool FinderPattern::aboutEquals(float estimatedRow, float estimatedCol, float estimatedModuleSize)
{
	if (cv::abs(estimatedRow - row) <= moduleSize && cv::abs(estimatedCol - col) <= moduleSize)
	{
		float moduleSizeDiff = cv::abs(moduleSize - estimatedModuleSize);
		return moduleSizeDiff <= 1.0f || moduleSizeDiff <= estimatedModuleSize;
	}
	return false;
}

void FinderPattern::combineEstimate(float estimatedRow, float estimatedCol, float estimatedModuleSize)
{
	row = (count * row + estimatedRow) / (count + 1);
	col = (count * col + estimatedCol) / (count + 1);
	moduleSize = (count * moduleSize + estimatedModuleSize) / (count + 1);
	count++;
}
