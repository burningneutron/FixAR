#include "RobustEstimator.h"
#include "Utils.h"

#include <time.h>

using namespace std;
using namespace cv;

namespace ar
{
#define SAMPLE_COLINE_THRESH 3

	// Point P3
	// Line P1---P2
	// Ref: http://paulbourke.net/geometry/pointline/
	static float pointToLineDistance2( Point2f p1, Point2f p2, Point2f p3)
	{
		float u =  ( (p3.x - p1.x)*(p2.x - p1.x) + (p3.y - p1.y)*(p2.y - p1.y) ) / ( AR_SQUARE(p2.x - p1.x) + AR_SQUARE(p2.y - p1.y) );
		float x = p1.x + u*(p2.x - p1.x);
		float y = p1.y + u*(p2.y - p1.y);

		return AR_SQUARE(p3.x - x) + AR_SQUARE(p3.y - y);
	}

	static double factorial(int n)
	{
		double t=1;
		for (int i=n;i>1;i--)
			t*=i;
		return t;
	}

	static double binomial_distribution(int n,double p,int r)
	{
		return factorial(n)/(factorial(n-r)*factorial(r))*pow(p,r)*pow(1-p,n-r);
	}

	static inline int Imin(int m, int n, double beta) {
			const double mu = n*beta;
			const double sigma = sqrt(n*beta*(1-beta));
			// Imin(n) (equation (8) can then be obtained with the Chi-squared test with P=2*psi=0.10 (Chi2=2.706)
			return (int)ceil(m + mu + sigma*sqrt(2.706));
	}

	ModelBase::~ModelBase()
	{

	}

	ModelBase* PerspectiveModel::clone()
	{
		PerspectiveModel* rtv = new PerspectiveModel;
		*rtv = *this;

		return rtv;
	}

	void PerspectiveModel::assign(const ModelBase& other)
	{
		const PerspectiveModel &_other = dynamic_cast<const PerspectiveModel&>(other);

		if( this != &_other ) *this = _other;
	}

	static inline bool isColine(const KeyPoint &p1, const KeyPoint &p2, const KeyPoint &p3)
	{
		double dx1 = p2.pt.x - p1.pt.x;
		double dy1 = p2.pt.y - p1.pt.y;

		double dx2 = p3.pt.x - p1.pt.x;
		double dy2 = p3.pt.y - p1.pt.y;
		if( fabs(dx2*dy1 - dy2*dx1) <= FLT_EPSILON*(fabs(dx1) + fabs(dy1) + fabs(dx2) + fabs(dy2)) )
			return true;
		else
			return false;
	}

	bool PerspectiveModel::isGoodSample(const std::vector< cv::KeyPoint > &minimalSetFrom, const std::vector< cv::KeyPoint > &minimalSetTo) const
	{
		assert(minimalSetFrom.size() == 4 && minimalSetTo.size() == 4);
		if( isColine(minimalSetFrom[0], minimalSetFrom[1], minimalSetFrom[2]) ) return false;
		if( isColine(minimalSetFrom[0], minimalSetFrom[1], minimalSetFrom[3]) ) return false;
		if( isColine(minimalSetFrom[0], minimalSetFrom[2], minimalSetFrom[3]) ) return false;
		if( isColine(minimalSetFrom[1], minimalSetFrom[2], minimalSetFrom[3]) ) return false;

		if( isColine(minimalSetTo[0], minimalSetTo[1], minimalSetTo[2]) ) return false;
		if( isColine(minimalSetTo[0], minimalSetTo[1], minimalSetTo[3]) ) return false;
		if( isColine(minimalSetTo[0], minimalSetTo[2], minimalSetTo[3]) ) return false;
		if( isColine(minimalSetTo[1], minimalSetTo[2], minimalSetTo[3]) ) return false;

		return true;		
	}

	bool PerspectiveModel::fit(const std::vector< cv::KeyPoint > &from, const std::vector< cv::KeyPoint > &to)
	{
		Mat m1 = Mat(Size(2, from.size()), CV_64FC1);
		Mat m2 = Mat(Size(2, to.size()), CV_64FC1);
		for( int i = 0; i < m1.rows; i++ ){
			m1.ptr<double>(i)[0] = from[i].pt.x;
			m1.ptr<double>(i)[1] = from[i].pt.y;

			m2.ptr<double>(i)[0] = to[i].pt.x;
			m2.ptr<double>(i)[1] = to[i].pt.y;
		}

		// codes are copied from opencv CvHomographyEstimator::runKernel
		int i, count = m1.rows;
		const CvPoint2D64f* M = (const CvPoint2D64f*)m1.data;
		const CvPoint2D64f* m = (const CvPoint2D64f*)m2.data;

		double LtL[9][9], W[9][1], V[9][9];
		CvMat _LtL = cvMat( 9, 9, CV_64F, LtL );
		CvMat matW = cvMat( 9, 1, CV_64F, W );
		CvMat matV = cvMat( 9, 9, CV_64F, V );
		CvMat _H0 = cvMat( 3, 3, CV_64F, V[8] );
		CvMat _Htemp = cvMat( 3, 3, CV_64F, V[7] );
		CvPoint2D64f cM={0,0}, cm={0,0}, sM={0,0}, sm={0,0};

		for( i = 0; i < count; i++ )
		{
			cm.x += m[i].x; cm.y += m[i].y;
			cM.x += M[i].x; cM.y += M[i].y;
		}

		cm.x /= count; cm.y /= count;
		cM.x /= count; cM.y /= count;

		for( i = 0; i < count; i++ )
		{
			sm.x += fabs(m[i].x - cm.x);
			sm.y += fabs(m[i].y - cm.y);
			sM.x += fabs(M[i].x - cM.x);
			sM.y += fabs(M[i].y - cM.y);
		}

		if( fabs(sm.x) < DBL_EPSILON || fabs(sm.y) < DBL_EPSILON ||
			fabs(sM.x) < DBL_EPSILON || fabs(sM.y) < DBL_EPSILON ){
				success_ = false;
				return false;
		}
		sm.x = count/sm.x; sm.y = count/sm.y;
		sM.x = count/sM.x; sM.y = count/sM.y;

		double invHnorm[9] = { 1./sm.x, 0, cm.x, 0, 1./sm.y, cm.y, 0, 0, 1 };
		double Hnorm2[9] = { sM.x, 0, -cM.x*sM.x, 0, sM.y, -cM.y*sM.y, 0, 0, 1 };
		CvMat _invHnorm = cvMat( 3, 3, CV_64FC1, invHnorm );
		CvMat _Hnorm2 = cvMat( 3, 3, CV_64FC1, Hnorm2 );

		cvZero( &_LtL );
		for( i = 0; i < count; i++ )
		{
			double x = (m[i].x - cm.x)*sm.x, y = (m[i].y - cm.y)*sm.y;
			double X = (M[i].x - cM.x)*sM.x, Y = (M[i].y - cM.y)*sM.y;
			double Lx[] = { X, Y, 1, 0, 0, 0, -x*X, -x*Y, -x };
			double Ly[] = { 0, 0, 0, X, Y, 1, -y*X, -y*Y, -y };
			int j, k;
			for( j = 0; j < 9; j++ )
				for( k = j; k < 9; k++ )
					LtL[j][k] += Lx[j]*Lx[k] + Ly[j]*Ly[k];
		}
		cvCompleteSymm( &_LtL );

		//cvSVD( &_LtL, &matW, 0, &matV, CV_SVD_MODIFY_A + CV_SVD_V_T );
		cvEigenVV( &_LtL, &matV, &matW );
		cvMatMul( &_invHnorm, &_H0, &_Htemp );
		cvMatMul( &_Htemp, &_Hnorm2, &_H0 );
		//cvConvertScale( &_H0, H, 1./_H0.data.db[8] );

		for( int i = 0; i < 9; i++ ){
			H_[i] = (float) ( _H0.data.db[i] / _H0.data.db[8] );
			//cout << H_[i] << endl;
		}

		success_ = true;
		return true;
	}

	void PerspectiveModel::repojErr(
		const std::vector< cv::KeyPoint > &from, const std::vector< cv::KeyPoint > &to,
		std::vector< float > &projErr) const
	{
		int count = (int) from.size();
		projErr.resize(count);
		
		for( int i = 0; i < count; i++ )
		{
			double ww = 1./(H_[6]*from[i].pt.x + H_[7]*from[i].pt.y + 1.);
			double dx = (H_[0]*from[i].pt.x + H_[1]*from[i].pt.y + H_[2])*ww - to[i].pt.x;
			double dy = (H_[3]*from[i].pt.x + H_[4]*from[i].pt.y + H_[5])*ww - to[i].pt.y;
			projErr[i] = (float)(dx*dx + dy*dy);
			//cout << projErr[i] << endl;
		}
	}

	int PerspectiveModel::findInlier(
		const std::vector< cv::KeyPoint > &from, const std::vector< cv::KeyPoint > &to,
		std::vector< int > &inlier, float thresh) const
	{
		int count = (int) from.size();
		vector< float > projErr;

		repojErr(from, to, projErr);

		float _thresh = thresh*thresh;

		inlier.clear();
		for( int i = 0; i < count; i++ ){
			if(projErr[i] < _thresh) inlier.push_back(i);
		}

		return inlier.size();
	}

	RobustEstimatorBase::~RobustEstimatorBase()
	{

	}

	bool RobustEstimatorRansac::estimate(
		const std::vector< cv::KeyPoint > &from, const std::vector< cv::KeyPoint > &to,
		ModelBase &model,
		std::vector< int > &inlier,
		float thresh,
		int maxIterNum, float confidence) const
	{
		assert(from.size() == to.size());
		assert(maxIterNum > 0);

		int count = (int) from.size();
		if( count < 4 ) return false;

		int iter;

		// ʹ�ù̶���������ӣ��Ա�֤ȷ���ԵĽ��
		srand(0);

		ModelBase *bestModel = model.clone();
		int bestInlierCount = -1;
		vector<int> bestInlier;

		for( iter = 0; iter < maxIterNum; iter++ ){
			vector< KeyPoint > minmalSetFrom, minmalSetTo;

			// 1: sample
			bool sampleSuccess = sample(from, to, model, minmalSetFrom, minmalSetTo);
			
			if( !sampleSuccess ){
				if( iter == 0 ) return false;
				else			continue;
			}
			
			// 2: fit model
			bool fitSuccess = model.fit(minmalSetFrom, minmalSetTo);
			if(!fitSuccess) continue;

			// 3: find inlier
			vector<int> curInlier;
			int curInlierCount = 0;
			curInlierCount = model.findInlier(from, to, curInlier, thresh);

			// 4: save best model
			double curInlierRatio = curInlierCount / (double)count;
			if( curInlierCount > bestInlierCount ){
				bestModel->assign(model);
				bestInlier = curInlier;
				bestInlierCount = curInlierCount;

				// 5: update max iter num
				maxIterNum = updateMaxIterNum(confidence, 1 - curInlierRatio, model.getModelPoints(), maxIterNum);
			}
		}

		// 6: refine
		for( int k = 0; k < 2; k++){
			model.assign(*bestModel);
			vector<KeyPoint> _from, _to;
			for( int i = 0; i < bestInlierCount; i++ ){
				_from.push_back(from[bestInlier[i]]);
				_to.push_back(to[bestInlier[i]]);
			}

			bool fitSuccess = model.fit(_from, _to);
			if( !fitSuccess ) continue;

			bestInlierCount = model.findInlier(from, to, bestInlier, thresh);
			bestModel->assign(model);
		}

		model.assign(*bestModel);
		inlier = bestInlier;
		delete bestModel;

		return !model.empty();
	}

	bool RobustEstimatorRansac::sample(
		const std::vector< cv::KeyPoint > &from, const std::vector< cv::KeyPoint > &to,
		ModelBase &model,
		std::vector< cv::KeyPoint > &minimalSetFrom, std::vector< cv::KeyPoint > &minimalSetTo) const
	{
		assert(from.size() == to.size());
		int ptCount = (int) from.size();

		int modelPoints = model.getModelPoints();
		if( ptCount < modelPoints ) return false;

		int maxTry = 300;
		
		bool isGood = false;
		int curTry = 0;
		while(curTry++ < maxTry){
			minimalSetFrom.clear();
			minimalSetTo.clear();

			for( int i = 0; i < modelPoints; i++ ){
				int idx = rand()%ptCount;
				minimalSetFrom.push_back(from[idx]);
				minimalSetTo.push_back(to[idx]);
			}

			if( model.isGoodSample(minimalSetFrom, minimalSetTo) ){
				isGood = true;
				break;
			}
		}

		if( isGood ) return true;
		else{
			minimalSetFrom.clear();
			minimalSetTo.clear();
			return false;
		}
	}

	int RobustEstimatorRansac::updateMaxIterNum(double p/*confidence*/, double ep/*outlier ratio*/, int model_points, int max_iters) const
	{
		// copied from opencv
		// Ref: http://en.wikipedia.org/wiki/RANSAC
		// maxIter = log((1-confidence)) / log(1 - (1-inlier_ratio)^model_point)
		if( model_points <= 0 )
			CV_Error( CV_StsOutOfRange, "the number of model points should be positive" );

		p = MAX(p, 0.);
		p = MIN(p, 1.);
		ep = MAX(ep, 0.);
		ep = MIN(ep, 1.);

		// avoid inf's & nan's
		double num = MAX(1. - p, DBL_MIN);
		double denom = 1. - pow(1. - ep,model_points);
		if( denom < DBL_MIN )
			return 0;

		num = log(num);
		denom = log(denom);

		return denom >= 0 || -num >= max_iters*(-denom) ? max_iters : cvRound(num/denom);
	}

	bool RobustEstimatorProsac::estimate(
		const std::vector< cv::KeyPoint > &from, const std::vector< cv::KeyPoint > &to,
		ModelBase &model,
		std::vector< int > &inlier,
		float thresh,
		int maxIterNum, float confidence) const
	{
		assert(from.size() == to.size());
		int count = (int) from.size();

		// Initialize some PROSAC constants
		int T_N = 200000;
		int N = (int) from.size ();
		int m = model.getModelPoints();
		float T_n = (float) T_N;
		for (int i = 0; i < m; ++i)
			T_n *= (float)(m - i) / (float)(N - i);
		float T_prime_n = 1;
		int I_N_best = 0;
		int n = m;

		// Define the n_Start coefficients from Section 2.2
		float n_star = (float) N;
		int I_n_star = 0;
		float epsilon_n_star = (float)I_n_star / n_star;
		int k_n_star = T_N;

		// Compute the I_n_star_min of Equation 8
		std::vector<int> I_n_star_min (N);

		// Initialize the usual RANSAC parameters
		int iterations = 0;

		std::vector<int> inliers;
		std::vector<int> selection;

		std::vector<int> indices(count);
		for( int i = 0; i < count; i++ ){
			indices[i] = i;
		}

		// We will increase the pool so the indices_ vector can only contain m elements at first
		std::vector<int> index_pool;
		index_pool.reserve (N);
		for (int i = 0; i < n; ++i)
			index_pool.push_back (indices.operator[](i)); // ????

		vector<int> bestInliers;
		ModelBase* bestModel = model.clone();
		
		// ʹ�ù̶���������ӣ��Ա�֤ȷ���ԵĽ��
		srand(0);

		// Iterate
		while ((int)iterations < k_n_star)
		{
			// Choose the samples

			// Step 1
			// According to Equation 5 in the text text, not the algorithm
			if ((iterations == T_prime_n) && (n < n_star))
			{
				// Increase the pool
				++n;
				if (n >= N)
					break;
				index_pool.push_back (indices.at(n - 1));
				// Update other variables
				float T_n_minus_1 = T_n;
				T_n *= (float)(n + 1) / (float)(n + 1 - m);
				T_prime_n += ceil(T_n - T_n_minus_1);
			}

			// Step 2
			vector<KeyPoint> minimalSetFrom, minimalSetTo;
			indices.swap (index_pool);
			selection.clear ();
			sample(from, to, model, index_pool, selection);
			//sac_model_->getSamples (iterations, selection); /// ???????????????????????????????????????
			if (T_prime_n < iterations)
			{
				selection.pop_back ();
				selection.push_back (indices.at(n - 1));
			}

			// Make sure we use the right indices for testing
			indices.swap (index_pool);

			if (selection.empty ())
			{
				break;
			}

			// Search for inliers in the point cloud for the current model
			for( int i = 0; i < (int) selection.size(); i++ ){
				minimalSetFrom.push_back(from[selection[i]]);
				minimalSetTo.push_back(to[selection[i]]);
			}

			bool success = model.fit(minimalSetFrom, minimalSetTo);

			if (!success)
			{
				++iterations;
				continue;
			}

			// Select the inliers that are within threshold_ from the model
			inliers.clear ();
			model.findInlier(from, to, inliers, thresh);

			int I_N = inliers.size ();

			// If we find more inliers than before
			if (I_N > I_N_best)
			{
				I_N_best = I_N;

				// Save the current model/inlier/coefficients selection as being the best so far
				bestInliers = inliers;
				bestModel->assign(model);
				//model_ = selection;

				// We estimate I_n_star for different possible values of n_star by using the inliers
				std::sort (inliers.begin (), inliers.end ());

				// Try to find a better n_star
				// We minimize k_n_star and therefore maximize epsilon_n_star = I_n_star / n_star
				int possible_n_star_best = N, I_possible_n_star_best = I_N;
				float epsilon_possible_n_star_best = (float)I_possible_n_star_best / possible_n_star_best;

				// We only need to compute possible better epsilon_n_star for when _n is just about to be removed an inlier
				int I_possible_n_star = I_N;
				for (std::vector<int>::const_reverse_iterator last_inlier = inliers.rbegin (); last_inlier != inliers.rend (); ++last_inlier, --I_possible_n_star)
				{
					// The best possible_n_star for a given I_possible_n_star is the index of the last inlier
					int possible_n_star = (*last_inlier) + 1;
					if (possible_n_star <= m)
						break;

					// If we find a better epsilon_n_star
					float epsilon_possible_n_star = (float)I_possible_n_star / possible_n_star;
					// Make sure we have a better epsilon_possible_n_star
					if ((epsilon_possible_n_star > epsilon_n_star) && (epsilon_possible_n_star > epsilon_possible_n_star_best))
					{
						// Typo in Equation 7, not (n-m choose i-m) but (n choose i-m)
						//int	I_possible_n_star_min = m
						//	+ ceil (quantile (complement (binomial_distribution<float>(possible_n_star, 0.1), 0.05)));
						int	I_possible_n_star_min = Imin(m, possible_n_star, 0.01);

						// If Equation 9 is not verified, exit
						if (I_possible_n_star < I_possible_n_star_min)
							break;

						possible_n_star_best = possible_n_star;
						I_possible_n_star_best = I_possible_n_star;
						epsilon_possible_n_star_best = epsilon_possible_n_star;
					}
				}

				// Check if we get a better epsilon
				if (epsilon_possible_n_star_best > epsilon_n_star)
				{
					// update the best value
					epsilon_n_star = epsilon_possible_n_star_best;

					// Compute the new k_n_star
					float bottom_log = 1 - std::pow (epsilon_n_star, static_cast<float>(m));
					if (bottom_log == 0)
						k_n_star = 1;
					else if (bottom_log == 1)
						k_n_star = (int) T_N;
					else
						k_n_star = (int)ceil (log(0.05) / log (bottom_log));
					// It seems weird to have very few iterations, so do have a few (totally empirical)
					k_n_star = (std::max)(k_n_star, 2 * m);
				}
			}

			++iterations;
			if (iterations > maxIterNum)
			{
				break;
			}
		}

		if (bestModel->empty ())
		{
			inlier.clear ();
			return (false);
		}

		// Get the set of inliers that correspond to the best model found so far
		model.assign(*bestModel);
		model.findInlier(from, to, inlier, thresh);
		delete []bestModel;

		return (true);
	}

	bool RobustEstimatorProsac::sample(
		const std::vector< cv::KeyPoint > &from, const std::vector< cv::KeyPoint > &to,
		ModelBase &model,
		const std::vector<int> &index_pool, std::vector<int> &selection) const
	{
		assert(from.size() == to.size());
		int ptCount = (int) index_pool.size();

		int modelPoints = model.getModelPoints();
		if( ptCount < modelPoints ) return false;

		vector<KeyPoint> minimalSetFrom, minimalSetTo;
		int maxTry = 100;

		bool isGood = false;
		int curTry = 0;
		while(curTry++ < maxTry){
			minimalSetFrom.clear();
			minimalSetTo.clear();
			selection.clear();

			for( int i = 0; i < modelPoints; i++ ){
				int idx = index_pool[rand()%ptCount];
				selection.push_back(idx);

				minimalSetFrom.push_back(from[idx]);
				minimalSetTo.push_back(to[idx]);
			}

			if( model.isGoodSample(minimalSetFrom, minimalSetTo) ){
				isGood = true;
				break;
			}
		}

		if( isGood ) return true;
		else{
			selection.clear();
			return false;
		}
	}

}
