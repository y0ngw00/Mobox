#ifndef __ENV_H__
#define __ENV_H__
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <utility>
namespace Eigen
{
	typedef Eigen::Matrix<bool, Eigen::Dynamic, 1> VectorXb;
}
class Environment;

class Env
{
public:
	Env();

	bool isEnableGoal();

	int getDimState();
	int getDimStateAMP();
	int getNumTotalLabel();
	int getDimAction();

	void reset();
	void step(const Eigen::VectorXd& action);
	
	double getRewardGoal();
	const Eigen::VectorXd& getState();
	const Eigen::VectorXd& getStateAMP();
	const bool& inspectEndOfEpisode();

	const Eigen::MatrixXd& getStatesAMPExpert();
private:
	Environment* mEnv;

	Eigen::VectorXd mState, mStateAMP;

	double mRewardGoal;
	bool mEOE;

	Eigen::MatrixXd mStatesAMPExpert;
};
#endif