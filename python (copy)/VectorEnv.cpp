#include "VectorEnv.h"
#include "Environment.h"
#include <functional>
#include <omp.h>
VectorEnv::
VectorEnv(int num_envs)
	:mNumEnvs(num_envs)
{
	omp_set_num_threads(mNumEnvs);
	mEnvs.resize(mNumEnvs, nullptr);
#pragma omp parallel for
	for(int i=0;i<mNumEnvs;i++)
	{	
		mEnvs[i] = new Environment();
	}

	mStates.resize(mNumEnvs,this->getDimState());
	mStatefs.resize(mNumEnvs);
	mRewards.resize(mNumEnvs);
	mRewardfs.resize(mNumEnvs);
	mEOEs.resize(mNumEnvs);
	mSleepf.resize(mNumEnvs);
}
int
VectorEnv::
getNumEnvs()
{
	return mNumEnvs;
}
int
VectorEnv::
getDimState()
{
	return mEnvs[0]->getDimState();
}
int
VectorEnv::
getDimAction()
{
	return mEnvs[0]->getDimAction();
}
int
VectorEnv::
getDimStatef()
{
	return mEnvs[0]->getDimStatef();
}
int
VectorEnv::
getDimActionf()
{
	return mEnvs[0]->getDimActionf();
}
void
VectorEnv::
reset(int id)
{
	mEnvs[id]->reset();
}
void
VectorEnv::
resets()
{
#pragma omp parallel for
	for(int i=0;i<mNumEnvs;i++)
	{	
		mEnvs[i]->reset();
	}
}
void
VectorEnv::
steps(const Eigen::MatrixXd& action, const std::vector<Eigen::VectorXd>& actionf)
{
	
	int cols = this->getDimActionf();

	auto reshape = [&](const Eigen::VectorXd& a)->Eigen::MatrixXd{
		int rows = a.rows()/cols;
		Eigen::MatrixXd reshaped(rows, cols);
		for(int j=0;j<rows;j++)
			reshaped.row(j) = a.segment(j*cols, cols);
		return reshaped;
	};

#pragma omp parallel for
	for(int i=0;i<mNumEnvs;i++)
	{
		mEnvs[i]->step(action.row(i), reshape(actionf[i]));
	}
	
}
const Eigen::MatrixXd&
VectorEnv::
getStates()
{
#pragma omp parallel for
	for(int i=0;i<mNumEnvs;i++)
	{
		mStates.row(i) = mEnvs[i]->getState();
	}

	return mStates;
}
std::vector<Eigen::MatrixXd>
VectorEnv::
getStatefs()
{
#pragma omp parallel for
	for(int i=0;i<mNumEnvs;i++)
	{
		mStatefs[i] = mEnvs[i]->getStatef();
	}
	return mStatefs;
}
const Eigen::VectorXd&
VectorEnv::
getRewards()
{
#pragma omp parallel for
	for(int i=0;i<mNumEnvs;i++)
	{
		mRewards[i] = mEnvs[i]->getReward()["r"];
		mRewardfs[i] = mEnvs[i]->getReward()["r_force"];
	}
	return mRewards;
}

const Eigen::VectorXd&
VectorEnv::
getRewardfs()
{
	return mRewardfs;
}


const Eigen::VectorXb&
VectorEnv::
isSleepf()
{
#pragma omp parallel for
	for(int i=0;i<mNumEnvs;i++)
	{
		mSleepf[i] = mEnvs[i]->isSleepf();
	}

	return mSleepf;
}
const Eigen::VectorXb&
VectorEnv::
inspectEndOfEpisodes()
{
#pragma omp parallel for
	for(int i=0;i<mNumEnvs;i++)
	{
		mEOEs[i] = mEnvs[i]->inspectEndOfEpisode();
	}

	return mEOEs;
}
void
VectorEnv::
syncEnvs()
{
	int n = mEnvs[0]->getInitialStateDistribution()->getValue().rows();
	
	Eigen::ArrayXXd initial_state_distributions(mNumEnvs, n);
	for(int i=0;i<mNumEnvs;i++)
		initial_state_distributions.row(i) = mEnvs[i]->getInitialStateDistribution()->getValue();

	Eigen::ArrayXd mean_value = initial_state_distributions.colwise().mean();
	for(int i=0;i<mNumEnvs;i++)
		mEnvs[i]->getInitialStateDistribution()->setValue(mean_value);

}
void
VectorEnv::
setKinematics(bool kin)
{
	for(int i=0;i<mNumEnvs;i++)
	{
		mEnvs[i]->setKinematics(kin);
	}
}

const Eigen::ArrayXd&
VectorEnv::
getInitialStateDistribution()
{
	return mEnvs[0]->getInitialStateDistribution()->getValue();
}
namespace py = pybind11;

PYBIND11_MODULE(pycomcon, m){
	py::class_<VectorEnv>(m, "vector_env")
		.def(py::init<int>())
		.def("get_num_envs", &VectorEnv::getNumEnvs)
		.def("get_dim_state", &VectorEnv::getDimState)
		.def("get_dim_action", &VectorEnv::getDimAction)
		.def("get_dim_statef", &VectorEnv::getDimStatef)
		.def("get_dim_actionf", &VectorEnv::getDimActionf)
		.def("reset", &VectorEnv::reset)
		.def("resets", &VectorEnv::resets)
		.def("steps", &VectorEnv::steps)
		.def("get_states", &VectorEnv::getStates)
		.def("get_statefs", &VectorEnv::getStatefs)
		.def("get_rewards", &VectorEnv::getRewards)
		.def("get_rewardfs", &VectorEnv::getRewardfs)
		.def("sync_envs", &VectorEnv::syncEnvs)
		.def("set_kinematics", &VectorEnv::setKinematics)
		.def("get_initial_state_distribution", &VectorEnv::getInitialStateDistribution)
		.def("is_sleepf", &VectorEnv::isSleepf)
		.def("inspect_end_of_episodes", &VectorEnv::inspectEndOfEpisodes);
	// py::class_<Environment>(m, "env")
	// 	.def(py::init<>())
	// 	.def("reset",&Environment::reset)
	// 	.def("step",&Environment::step)
	// 	.def("get_state",&Environment::getState)
	// 	.def("get_reward",&Environment::getReward)
	// 	.def("get_info",&Environment::getInfo)
	// 	.def("inspect_end_of_episode",&Environment::inspectEndOfEpisode)
	// 	.def("get_dim_state",&Environment::getDimState)
	// 	.def("get_dim_action",&Environment::getDimAction)
	// 	.def("get_average_force_reward",&Environment::getAverageForceReward)
	// 	.def("get_force_distribution",&Environment::getForceDistribution)
	// 	.def("set_force_distribution",&Environment::setForceDistribution)
	// 	.def("get_ball_distribution",&Environment::getBallDistribution)
	// 	.def("set_ball_distribution",&Environment::setBallDistribution)
	// 	.def("set_current_force_boundary",&Environment::setCurrentForceBoundary);
}