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
	mStatesAMP.resize(mNumEnvs, this->getDimStateAMP());
	
	mEOEs.resize(mNumEnvs);
	mRewardGoals.resize(mNumEnvs);
}
bool
VectorEnv::
isEnableGoal()
{
	return mEnvs[0]->isEnableGoal();
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
getDimStateAMP()
{
	return mEnvs[0]->getDimStateAMP();
}
int
VectorEnv::
getDimAction()
{
	return mEnvs[0]->getDimAction();
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
steps(const Eigen::MatrixXd& action)
{
#pragma omp parallel for
	for(int i=0;i<mNumEnvs;i++)
	{
		mEnvs[i]->step(action.row(i));
	}
	
}
const Eigen::VectorXd&
VectorEnv::
getRewardGoals()
{
	for(int i=0;i<mNumEnvs;i++)
		mRewardGoals[i] = mEnvs[i]->getRewardGoal();
	return mRewardGoals;
}
const Eigen::MatrixXd&
VectorEnv::
getStates()
{
	for(int i=0;i<mNumEnvs;i++)
		mStates.row(i) = mEnvs[i]->getState();
	return mStates;
}

const Eigen::MatrixXd&
VectorEnv::
getStatesAMP()
{
	for(int i=0;i<mNumEnvs;i++)
		mStatesAMP.row(i) = mEnvs[i]->getStateAMP();
	return mStatesAMP;
}
const Eigen::VectorXb&
VectorEnv::
inspectEndOfEpisodes()
{
	for(int i=0;i<mNumEnvs;i++)
		mEOEs[i] = mEnvs[i]->inspectEndOfEpisode();

	return mEOEs;
}
const Eigen::MatrixXd&
VectorEnv::
getStatesAMPExpert()
{
	mStatesAMPExpert = mEnvs[0]->getStateAMPExpert();
	return mStatesAMPExpert;
}


namespace py = pybind11;

PYBIND11_MODULE(pycomcon, m){
	py::class_<VectorEnv>(m, "vector_env")
		.def(py::init<int>())
		.def("is_enable_goal", &VectorEnv::isEnableGoal)
		.def("get_num_envs", &VectorEnv::getNumEnvs)
		.def("get_dim_state", &VectorEnv::getDimState)
		.def("get_dim_state_AMP", &VectorEnv::getDimStateAMP)
		.def("get_dim_action", &VectorEnv::getDimAction)
		.def("reset", &VectorEnv::reset)
		.def("resets", &VectorEnv::resets)
		.def("steps", &VectorEnv::steps)
		.def("get_reward_goals", &VectorEnv::getRewardGoals)
		.def("get_states", &VectorEnv::getStates)
		.def("get_states_AMP", &VectorEnv::getStatesAMP)
		.def("get_states_AMP_expert", &VectorEnv::getStatesAMPExpert)
		.def("inspect_end_of_episodes", &VectorEnv::inspectEndOfEpisodes);

		// .def(py::init<int>())
		// .def("get_num_envs", &VectorEnv::getNumEnvs)
		// .def("get_dim_state0", &VectorEnv::getDimState0)
		// .def("get_dim_action0", &VectorEnv::getDimAction0)
		// .def("get_dim_state1", &VectorEnv::getDimState1)
		// .def("get_dim_action1", &VectorEnv::getDimAction1)
		// .def("reset", &VectorEnv::reset)
		// .def("resets", &VectorEnv::resets)
		// .def("steps", &VectorEnv::steps)
		// .def("get_states", &VectorEnv::getStates)
		// .def("get_rewards", &VectorEnv::getRewards)
		// .def("inspect_end_of_episodes", &VectorEnv::inspectEndOfEpisodes)
		// .def("is_sleeps", &VectorEnv::isSleeps)
		// .def("sync_envs", &VectorEnv::syncEnvs)
		// .def("set_kinematics", &VectorEnv::setKinematics);
}