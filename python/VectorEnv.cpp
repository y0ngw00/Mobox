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


Env::
Env()
{
	mEnv = new Environment();

	mState.resize(this->getDimState());
	mStateAMP.resize(this->getDimStateAMP());
	
	mEOE = false;
	mRewardGoal = 1.0;
}
bool
Env::
isEnableGoal()
{
	return mEnv->isEnableGoal();
}
int
Env::
getDimState()
{
	return mEnv->getDimState();
}
int
Env::
getDimStateAMP()
{
	return mEnv->getDimStateAMP();
}
int
Env::
getDimAction()
{
	return mEnv->getDimAction();
}
void
Env::
reset()
{
	mEnv->reset();
}
void
Env::
step(const Eigen::VectorXd& action)
{
	mEnv->step(action);
}
double
Env::
getRewardGoal()
{
	mRewardGoal = mEnv->getRewardGoal();

	return mRewardGoal;
}
const Eigen::VectorXd&
Env::
getState()
{
	mState = mEnv->getState();
	return mState;
}
const Eigen::VectorXd&
Env::
getStateAMP()
{
	mStateAMP = mEnv->getStateAMP();
	return mStateAMP;
}
const bool&
Env::
inspectEndOfEpisode()
{
	mEOE = mEnv->inspectEndOfEpisode();
	return mEOE;
}

const Eigen::MatrixXd&
Env::
getStatesAMPExpert()
{
	mStatesAMPExpert = mEnv->getStateAMPExpert();
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

	py::class_<Env>(m, "env")
		.def(py::init<>())
		.def("is_enable_goal", &Env::isEnableGoal)
		.def("get_dim_state", &Env::getDimState)
		.def("get_dim_state_AMP", &Env::getDimStateAMP)
		.def("get_dim_action", &Env::getDimAction)
		.def("reset", &Env::reset)
		.def("step", &Env::step)
		.def("get_reward_goal", &Env::getRewardGoal)
		.def("get_state", &Env::getState)
		.def("get_state_AMP", &Env::getStateAMP)
		.def("inspect_end_of_episode", &Env::inspectEndOfEpisode)
		.def("get_states_AMP_expert", &Env::getStatesAMPExpert);
}