#pragma once
class FinderPattern
{
public:
	inline FinderPattern(float row, float col, float estimatedModuleSize)
		: row(row), col(col), moduleSize(estimatedModuleSize), count(1) {}
	virtual ~FinderPattern() {}

	bool aboutEquals(float row, float col, float estimatedModuleSize);
	void combineEstimate(float row, float col, float estimatedModuleSize);

	inline float getModuleSize() const { return moduleSize; }
	inline float getRow() const { return row; }
	inline float getCol() const { return col; }
	inline int getCount() const { return count; }
private:
	float row;
	float col;
	float moduleSize;
	int count;
};

