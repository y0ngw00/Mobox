#ifndef __VECTOR_ENV_H__
#define __VECTOR_ENV_H__
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <tuple>
namespace Eigen
{
	typedef Eigen::Matrix<bool, Eigen::Dynamic, 1> VectorXb;
}
class Environment;
class VectorEnv
{
public:
	VectorEnv(int num_envs);

	int getNumEnvs();
	int getDimState();
	int getDimAction();
	int getDimStatef();
	int getDimActionf();

	void reset(int id);
	void resets();
	void steps(const Eigen::MatrixXd& action, const std::vector<Eigen::VectorXd>& actionf);
	const Eigen::MatrixXd& getStates();
	std::vector<Eigen::MatrixXd> getStatefs();

	const Eigen::VectorXd& getRewards();
	const Eigen::VectorXd& getRewardfs();

	const Eigen::VectorXb& isSleepf();
	const Eigen::VectorXb& inspectEndOfEpisodes();

	void syncEnvs();
	void setKinematics(bool kin);

	const Eigen::ArrayXd& getInitialStateDistribution();
private:
	std::vector<Environment*> mEnvs;
	Eigen::MatrixXd mStates;
	std::vector<Eigen::MatrixXd> mStatefs;
	Eigen::VectorXd mRewards;
	Eigen::VectorXd mRewardfs;
	Eigen::VectorXb mEOEs;
	Eigen::VectorXb mSleepf;
	int mNumEnvs;
};
#endif