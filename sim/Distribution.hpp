#ifndef __DISTRIBUTION_H__
#define __DISTRIBUTION_H__
#include <Eigen/Core>
#include <functional>
#include "dart/dart.hpp"
template<typename T>
class Distribution1D
{
public:
	Distribution1D(int n, const std::function<T(int)>& f, double momentum=0.9);

	T sample();
	void update(double val);
	const Eigen::ArrayXd& getValue(){return mValue;}
	void setValue(const Eigen::ArrayXd& value){mValue = value;}
private:
	Eigen::ArrayXd mValue;
	int mCursor;
	std::function<T(int)> mf;

	int mN;
	double mMom;
};
// int stride = 10;
// auto f = [=](int i, int j)->int{
// 	return i*stride + j;
// };

// Distribution2D<int> distribution(10,10,f);
// for(int i=0;i<100;i++)
// {
// 	std::cout<<distribution.sample()<<" ";
// }

// std::cout<<std::endl;
template<typename T>
class Distribution2D
{
public:
	Distribution2D(int n, int m, const std::function<T(int, int)>& f, double momentum=0.9);

	T sample();
	void update(double val);

	const Eigen::ArrayXXd& getValue(){return mValue;}
	void setValue(const Eigen::ArrayXXd& value){mValue = value;}
private:
	Eigen::ArrayXXd mValue;
	int mCursorX, mCursorY;
	std::function<T(int, int)> mf;

	int mN, mM;
	double mMom;
};



template<typename T>
Distribution1D<T>::Distribution1D(int n, const std::function<T(int)>& f, double momentum)
	:mN(n),mf(f),mMom(momentum),mCursor(-1)
{
	mValue.resize(mN);
	mValue.setZero();
}

template<typename T>
T Distribution1D<T>::sample()
{
	double mean = mValue.mean();
	double var = (mValue*mValue).mean() - mean*mean;
	double std = std::sqrt(var);

	if(std<1e-6)
		std = 1.0;
	Eigen::ArrayXd rho = (mValue - mean)/std;
	rho = (-rho).exp();
	rho /= rho.sum();

	double r = dart::math::Random::uniform<double>(0.0, 1.0);
	double p = 0.0;

	for(int i=0;i<mN;i++)
	{
		p += rho[i];
		if(r<=p){
			mCursor = i;
			break;
		}
	}

	return mf(mCursor);
}

template<typename T>
void Distribution1D<T>::update(double val)
{
	mValue[mCursor] = mMom*mValue[mCursor] + (1.0 - mMom)*val;
	mCursor = -1;
}


template<typename T>
Distribution2D<T>::
Distribution2D(int n, int m, const std::function<T(int, int)>& f, double momentum)
	:mN(n), mM(m), mf(f), mMom(momentum), mCursorX(-1), mCursorY(-1)
{
	mValue.resize(mN, mM);
	

	mValue.setZero();
	for(int i=0;i<mN*mM;i++)
		mValue(i/mM, i-(int)(i/mM)*mM) = i;
}

template<typename T>
T Distribution2D<T>::sample()
{
	double mean = mValue.mean();
	double var = (mValue*mValue).mean() - mean*mean;
	double std = std::sqrt(var);

	if(std<1e-6)
		std = 1.0;
	Eigen::ArrayXXd rho = (mValue - mean)/std;
	rho = (-rho).exp();
	rho /= rho.sum();
	double r = dart::math::Random::uniform<double>(0.0, 1.0);
	double p = 0.0;
	for(int i=0;i<mN*mM;i++)
	{
		int x = i/mM;
		int y = i-x*mM;

		p += rho(x, y);
		if(r<=p)
		{
			mCursorX = x;
			mCursorY = y;
			break;
		}
	}

	return mf(mCursorX, mCursorY);
}
template<typename T>
void Distribution2D<T>::update(double val)
{
	mValue(mCursorX,mCursorY) = mMom*mValue(mCursorX,mCursorY) + (1.0 - mMom)*val;
	mCursorX = -1;
	mCursorY = -1;
}


#endif