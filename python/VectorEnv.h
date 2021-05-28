#ifndef __VECTOR_ENV_H__
#define __VECTOR_ENV_H__
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
class VectorEnv
{
public:
	VectorEnv(int num_envs);
	
	bool isEnableGoal();
	
	int getNumEnvs();
	int getDimState();
	int getDimStateAMP();
	int getDimAction();

	void reset(int id);
	void resets();
	void steps(const Eigen::MatrixXd& action);
	
	const Eigen::VectorXd& getRewardGoals();
	const Eigen::MatrixXd& getStates();	
	const Eigen::MatrixXd& getStatesAMP();
	const Eigen::VectorXb& inspectEndOfEpisodes();

	const Eigen::MatrixXd& getStatesAMPExpert();

private:
	std::vector<Environment*> mEnvs;
	Eigen::MatrixXd mStates, mStatesAMP, mStatesAMPExpert;
	Eigen::VectorXd mRewardGoals;
	Eigen::VectorXb mEOEs;
	int mNumEnvs;
};

class Env
{
public:
	Env();

	bool isEnableGoal();

	int getDimState();
	int getDimStateAMP();
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