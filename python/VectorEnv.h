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

	int getNumEnvs();
	int getDimState0();
	int getDimAction0();
	int getDimState1();
	int getDimAction1();

	void reset(int id);
	void resets();
	void steps(const Eigen::MatrixXd& action0, const Eigen::MatrixXd& action1);
	std::pair<Eigen::MatrixXd, Eigen::MatrixXd> getStates();
	std::pair<Eigen::VectorXd, Eigen::VectorXd> getRewards();
	const Eigen::VectorXb& inspectEndOfEpisodes();
	const Eigen::VectorXb& isSleeps();

	void syncEnvs();
	void setKinematics(bool kin);
private:
	std::vector<Environment*> mEnvs;
	Eigen::MatrixXd mStates0, mStates1;
	Eigen::VectorXd mRewards0, mRewards1;
	Eigen::VectorXb mEOEs;
	Eigen::VectorXb mSleeps;
	int mNumEnvs;
};
#endif