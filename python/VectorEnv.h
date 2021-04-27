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
	VectorEnv(int num_envs, bool enable_lower_upper_body);
	
	bool isEnableLowerUpperBody(){return mEnableLowerUpperBody;}
	int getNumEnvs();
	int getDimState();
	int getDimStateAMP();
	int getDimStateAMPLowerBody();
	int getDimStateAMPUpperBody();
	int getDimAction();

	void reset(int id);
	void resets();
	void steps(const Eigen::MatrixXd& action);
	
	const Eigen::VectorXd& getRewardGoals();
	const Eigen::MatrixXd& getStates();	
	const Eigen::MatrixXd& getStatesAMP();
	const Eigen::MatrixXd& getStatesAMPLowerBody();
	const Eigen::MatrixXd& getStatesAMPUpperBody();
	const Eigen::VectorXb& inspectEndOfEpisodes();

	const Eigen::MatrixXd& getStatesAMPExpert();
	const Eigen::MatrixXd& getStatesAMPExpertLowerBody();
	const Eigen::MatrixXd& getStatesAMPExpertUpperBody();
private:
	std::vector<Environment*> mEnvs;
	bool mEnableLowerUpperBody;
	Eigen::MatrixXd mStates, mStatesAMP, mStatesAMPExpert;
	Eigen::MatrixXd mStatesAMPLowerBody, mStatesAMPUpperBody, mStatesAMPExpertLowerBody, mStatesAMPExpertUpperBody;
	Eigen::VectorXd mRewardGoals;
	Eigen::VectorXb mEOEs;
	int mNumEnvs;
};

// 	int getNumEnvs();
// 	int getDimState0();
// 	int getDimAction0();
// 	int getDimState1();
// 	int getDimAction1();

// 	int getDimStateAMP();
// 	const Eigen::MatrixXd& getStatesAMP();

// 	void reset(int id);
// 	void resets();
// 	void steps(const Eigen::MatrixXd& action);
// 	const Eigen::MatrixXd& getStates();
// 	const Eigen::VectorXd& getRewards();
// 	const Eigen::VectorXb& inspectEndOfEpisodes();
// 	const Eigen::VectorXb& isSleeps();

// 	void syncEnvs();
// 	void setKinematics(bool kin);
// private:
// 	std::vector<Environment*> mEnvs;
// 	Eigen::MatrixXd mStates, mStatesAMP;
// 	Eigen::VectorXd mRewards;
// 	Eigen::VectorXb mEOEs;
// 	Eigen::VectorXb mSleeps;
// 	int mNumEnvs;
// };
#endif