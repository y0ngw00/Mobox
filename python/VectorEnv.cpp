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

	mStates0.resize(mNumEnvs,this->getDimState0());
	mStates1.resize(mNumEnvs,this->getDimState1());

	mRewards0.resize(mNumEnvs);
	mRewards1.resize(mNumEnvs);
	mEOEs.resize(mNumEnvs);
	mSleeps.resize(mNumEnvs);
}
int
VectorEnv::
getNumEnvs()
{
	return mNumEnvs;
}
int
VectorEnv::
getDimState0()
{
	return mEnvs[0]->getDimState0();
}
int
VectorEnv::
getDimAction0()
{
	return mEnvs[0]->getDimAction0();
}
int
VectorEnv::
getDimState1()
{
	return mEnvs[0]->getDimState1();
}
int
VectorEnv::
getDimAction1()
{
	return mEnvs[0]->getDimAction1();
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
steps(const Eigen::MatrixXd& action0, const Eigen::MatrixXd& action1)
{
#pragma omp parallel for
	for(int i=0;i<mNumEnvs;i++)
	{
		mEnvs[i]->step(action0.row(i), action1.row(i));
	}
	
}
std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
VectorEnv::
getStates()
{
#pragma omp parallel for
	for(int i=0;i<mNumEnvs;i++)
	{
		mStates0.row(i) = mEnvs[i]->getState0();
		mStates1.row(i) = mEnvs[i]->getState1();

	}
	return std::make_pair(mStates0, mStates1);
}
std::pair<Eigen::VectorXd, Eigen::VectorXd>
VectorEnv::
getRewards()
{
#pragma omp parallel for
	for(int i=0;i<mNumEnvs;i++)
	{
		mRewards0[i] = mEnvs[i]->getReward()["r"];
		mRewards1[i] = mEnvs[i]->getReward()["r_force"];
	}
	return std::make_pair(mRewards0, mRewards1);
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
const Eigen::VectorXb&
VectorEnv::
isSleeps()
{
#pragma omp parallel for
	for(int i=0;i<mNumEnvs;i++)
	{
		mSleeps[i] = mEnvs[i]->isSleep();
	}

	return mSleeps;
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

namespace py = pybind11;

PYBIND11_MODULE(pycomcon, m){
	py::class_<VectorEnv>(m, "vector_env")
		.def(py::init<int>())
		.def("get_num_envs", &VectorEnv::getNumEnvs)
		.def("get_dim_state0", &VectorEnv::getDimState0)
		.def("get_dim_action0", &VectorEnv::getDimAction0)
		.def("get_dim_state1", &VectorEnv::getDimState1)
		.def("get_dim_action1", &VectorEnv::getDimAction1)
		.def("reset", &VectorEnv::reset)
		.def("resets", &VectorEnv::resets)
		.def("steps", &VectorEnv::steps)
		.def("get_states", &VectorEnv::getStates)
		.def("get_rewards", &VectorEnv::getRewards)
		.def("inspect_end_of_episodes", &VectorEnv::inspectEndOfEpisodes)
		.def("is_sleeps", &VectorEnv::isSleeps)
		.def("sync_envs", &VectorEnv::syncEnvs)
		.def("set_kinematics", &VectorEnv::setKinematics);
}