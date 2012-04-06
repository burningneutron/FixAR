#include <vector>
#include <opencv2/opencv.hpp>

namespace ar
{
	int findHomographyRansac();

	class ModelBase
	{
	public:
		ModelBase(){}
		virtual ~ModelBase() = 0;

		virtual ModelBase* clone() = 0;

		virtual void assign(const ModelBase& other) = 0;

		virtual bool empty() const = 0;

		virtual int getModelPoints() const = 0;

		virtual bool isGoodSample(const std::vector< cv::KeyPoint > &minimalSetFrom, const std::vector< cv::KeyPoint > &minimalSetTo) const = 0;

		virtual bool fit(const std::vector< cv::KeyPoint > &from, const std::vector< cv::KeyPoint > &to) = 0;

		virtual void repojErr(
			const std::vector< cv::KeyPoint > &from, const std::vector< cv::KeyPoint > &to,
			std::vector< float > &projErr) const = 0;

		virtual int findInlier(
			const std::vector< cv::KeyPoint > &from, const std::vector< cv::KeyPoint > &to,
			std::vector< int > &inlier, float thresh) const = 0;
	};

	// Homography
	class PerspectiveModel: public ModelBase
	{
	public:
		PerspectiveModel()
		{
			memset(H_, 0, 9*sizeof(H_[0]));
			success_ = false;
		}
		virtual ~PerspectiveModel() {}

		virtual ModelBase* clone();

		virtual void assign(const ModelBase& other);

		virtual bool empty() const { return !success_; }

		virtual int getModelPoints() const { return 4; }

		virtual bool isGoodSample(const std::vector< cv::KeyPoint > &minimalSetFrom, const std::vector< cv::KeyPoint > &minimalSetTo) const;

		virtual bool fit(const std::vector< cv::KeyPoint > &from, const std::vector< cv::KeyPoint > &to);

		virtual void repojErr(
			const std::vector< cv::KeyPoint > &from, const std::vector< cv::KeyPoint > &to,
			std::vector< float > &projErr) const;

		virtual int findInlier(
			const std::vector< cv::KeyPoint > &from, const std::vector< cv::KeyPoint > &to,
			std::vector< int > &inlier, float thresh) const;

		float H_[9];
		bool success_;
	};

	class AffineModel: public ModelBase
	{

	};

	class LineModel: public ModelBase
	{

	};

	class RobustEstimatorBase
	{
	public:
		RobustEstimatorBase(){}
		virtual ~RobustEstimatorBase() = 0;

		virtual bool estimate(
			const std::vector< cv::KeyPoint > &from, const std::vector< cv::KeyPoint > &to,
			ModelBase &model,
			std::vector< int > &inlier,
			float thresh,
			int maxIterNum, float confidence) const = 0;
	};

	class RobustEstimatorRansac: public RobustEstimatorBase
	{
	public:
		RobustEstimatorRansac(){}
		virtual ~RobustEstimatorRansac() {}

		virtual bool estimate(
			const std::vector< cv::KeyPoint > &from, const std::vector< cv::KeyPoint > &to,
			ModelBase &model,
			std::vector< int > &inlier,
			float thresh,
			int maxIterNum, float confidence = 0.995f) const;
	private:
		bool sample(
			const std::vector< cv::KeyPoint > &from, const std::vector< cv::KeyPoint > &to,
			ModelBase &model,
			std::vector< cv::KeyPoint > &minimalSetFrom, std::vector< cv::KeyPoint > &minimalSetTo) const;
		int updateMaxIterNum(double p, double ep, int model_points, int max_iters) const;
	};

	class RobustEstimatorProsac: public RobustEstimatorBase
	{
	public:
		RobustEstimatorProsac(){}
		virtual ~RobustEstimatorProsac() {}

		virtual bool estimate(
			const std::vector< cv::KeyPoint > &from, const std::vector< cv::KeyPoint > &to,
			ModelBase &model,
			std::vector< int > &inlier,
			float thresh,
			int maxIterNum, float confidence = 0.995f) const;
	private:
		bool sample(
			const std::vector< cv::KeyPoint > &from, const std::vector< cv::KeyPoint > &to,
			ModelBase &model,
			const std::vector<int> &index_pool, std::vector<int> &selection) const;
	};

}
