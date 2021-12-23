#include "Env.h"
#include "Environment.h"
#include <functional>

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
getNumTotalLabel()
{
	return mEnv->getNumTotalLabel();
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

const Eigen::VectorXd&
Env::
getNumFrames()
{
	mNumFrames = mEnv->getNumFrames();
	return mNumFrames;
}

namespace py = pybind11;

PYBIND11_MODULE(pycomcon, m){
	py::class_<Env>(m, "env")
		.def(py::init<>())
		.def("is_enable_goal", &Env::isEnableGoal)
		.def("get_dim_state", &Env::getDimState)
		.def("get_dim_state_AMP", &Env::getDimStateAMP)
		.def("get_num_total_label", &Env::getNumTotalLabel)
		.def("get_dim_action", &Env::getDimAction)
		.def("reset", &Env::reset)
		.def("step", &Env::step)
		.def("get_reward_goal", &Env::getRewardGoal)
		.def("get_state", &Env::getState)
		.def("get_state_AMP", &Env::getStateAMP)
		.def("inspect_end_of_episode", &Env::inspectEndOfEpisode)
		.def("get_states_AMP_expert", &Env::getStatesAMPExpert)
		.def("get_num_frames", &Env::getNumFrames);
}