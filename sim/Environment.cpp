#include <functional>
#include <fstream>
#include <sstream>
#include "Environment.h"
#include "DARTUtils.h"
#include "MathUtils.h"
#include "Character.h"
#include "BVH.h"
#include "Motion.h"
#include "dart/collision/bullet/bullet.hpp"
#include "dart/collision/fcl/fcl.hpp"

using namespace dart;
using namespace dart::simulation;
using namespace dart::dynamics;

Environment::
Environment()
	:mWorld(std::make_shared<World>()),
	mControlHz(30),
	mSimulationHz(600),
	mElapsedFrame(0),
	mMaxElapsedFrame(300),
	mSimCharacter(nullptr),
	mKinCharacter(nullptr),
	mTargetSpeedMin(0.5),
	mTargetSpeedMax(3.0),
	mSharpTurnProb(0.01),
	mSpeedChangeProb(0.05),
	mHeightChangeProb(0.01),
	mMaxHeadingTurnRate(0.15),
	mTransitionProb(0.005),
	mRewardGoal(0.0),
	mEnableGoal(true)
{
	dart::math::Random::generateSeed(true);

	mSimCharacter = DARTUtils::buildFromFile(std::string(ROOT_DIR)+"/data/skel.xml");
	mKinCharacter = DARTUtils::buildFromFile(std::string(ROOT_DIR)+"/data/skel.xml");

	char buffer[100];
	std::ifstream txtread;
	std::vector<std::string> motion_lists;
	std::string txt_path = "/data/bvh/motionlist.txt";
	txtread.open(std::string(ROOT_DIR)+txt_path);
	if(!txtread.is_open()){
		std::cout<<"Text file does not exist from : "<< txt_path << std::endl;
		return;
	}
	while(txtread>>buffer) motion_lists.push_back(std::string(ROOT_DIR)+"/data/bvh/"+ std::string(buffer));
	txtread.close();


	mNumMotions = motion_lists.size();
	mDimLabel = 1;

	bool load_tree =false;
	for(auto bvh_path : motion_lists){
		BVH* bvh = new BVH(bvh_path);
		Motion* motion = new Motion(bvh);
		for(int j=0;j<bvh->getNumFrames();j++){
			motion->append(bvh->getPosition(j), bvh->getRotation(j),false);
			if(j>300) break;
		}
		if(bvh->getNumFrames() < 300) motion->repeatMotion(300, bvh);

		motion->computeVelocity();
		mMotions.emplace_back(motion);

		if(!load_tree){
			mSimCharacter->buildBVHIndices(motion->getBVH()->getNodeNames());
			mKinCharacter->buildBVHIndices(motion->getBVH()->getNodeNames());
			load_tree = true;			
		}

		// delete bvh;
		// delete motion;
	}

	// BVH* bvh = new BVH(std::string(ROOT_DIR)+"/data/bvh/walk_long.bvh");
	// 	Motion* motion = new Motion(bvh);
	// 	for(int j=0;j<bvh->getNumFrames();j++)
	// 		motion->append(bvh->getPosition(j), bvh->getRotation(j),false);

	// 	motion->computeVelocity();
	// 	mMotions.emplace_back(motion);
	// 			mSimCharacter->buildBVHIndices(motion->getBVH()->getNodeNames());
	// 		mKinCharacter->buildBVHIndices(motion->getBVH()->getNodeNames());



	double ground_height = this->computeGroundHeight();
	mGround = DARTUtils::createGround(ground_height);
	mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());
	mWorld->addSkeleton(mSimCharacter->getSkeleton());
	mWorld->addSkeleton(mGround);
	mWorld->setTimeStep(1.0/(double)mSimulationHz);
	mWorld->setGravity(Eigen::Vector3d(0,-9.81,0.0));
	mElapsedFrame = 0;
	mFrame = 0;

	mSimCharacter->getSkeleton()->setSelfCollisionCheck(true);
	mSimCharacter->getSkeleton()->setAdjacentBodyCheck(false);
	this->reset();

	mActionSpace = this->getActionSpace();
	mActionWeight = this->getActionWeight();

	this->mTargetHeightMin = 0.6;
	this->mTargetHeightMax = 2.0;
	mStateLabel = 0;

}

int
Environment::
getDimState()
{
	return this->getState().rows();
}
int
Environment::
getDimAction()
{
	int n = mSimCharacter->getSkeleton()->getNumDofs();
	return n-6;
}
int
Environment::
getDimStateAMP()
{
	return this->getStateAMP().rows();
}
int
Environment::
getDimStateLabel()
{
	return this->mDimLabel;
}
int
Environment::
getNumTotalLabel()
{
	return this->mNumMotions;
}

void
Environment::
reset(bool RSI)
{
	mContactEOE = false;
	mFrame = 0;
	mElapsedFrame = 0;

	int motion_num=0;
	if(RSI){
		motion_num = dart::math::Random::uniform<int>(0, this->mNumMotions-1);
		//mFrame = dart::math::Random::uniform<int>(0,motion->getNumFrames()-3);
		mStateLabel = motion_num;
	}
	else{
		motion_num = mStateLabel;
	}

	auto motion = mMotions[motion_num];
	mFrame = dart::math::Random::uniform<int>(0,motion->getNumFrames()-3);


	Eigen::Vector3d position = motion->getPosition(mFrame);
	Eigen::MatrixXd rotation = motion->getRotation(mFrame);
	Eigen::Vector3d linear_velocity = motion->getLinearVelocity(mFrame);
	Eigen::MatrixXd angular_velocity = motion->getAngularVelocity(mFrame);

	mSimCharacter->setPose(position, rotation, linear_velocity, angular_velocity);

	mSimCharacter->getSkeleton()->clearConstraintImpulses();
	mSimCharacter->getSkeleton()->clearInternalForces();
	mSimCharacter->getSkeleton()->clearExternalForces();

	mKinCharacter->setPose(position, rotation, linear_velocity, angular_velocity);

	mPrevPositions2 = mSimCharacter->getSkeleton()->getPositions(); 
	mPrevPositions = mSimCharacter->getSkeleton()->getPositions();
	mPrevCOM = mSimCharacter->getSkeleton()->getCOM();

	if(mEnableGoal){
		this->resetGoal();
		this->recordGoal();	
	}

	this->recordState();
}
void
Environment::
step(const Eigen::VectorXd& _action)
{
	Eigen::VectorXd action = this->convertToRealActionSpace(_action);

	auto sim_skel = mSimCharacter->getSkeleton();
	int num_sub_steps = mSimulationHz/mControlHz;

	auto target_pos = mSimCharacter->computeTargetPosition(action);

	for(int i=0;i<num_sub_steps;i++)
	{
		mSimCharacter->actuate(target_pos);
		mWorld->step();

		auto cr = mWorld->getConstraintSolver()->getLastCollisionResult();

		for(int j=0;j<cr.getNumContacts();j++)
		{
			auto contact = cr.getContact(j);
			auto shapeFrame1 = const_cast<dart::dynamics::ShapeFrame*>(contact.collisionObject1->getShapeFrame());
			auto shapeFrame2 = const_cast<dart::dynamics::ShapeFrame*>(contact.collisionObject2->getShapeFrame());

			auto bn1 = shapeFrame1->asShapeNode()->getBodyNodePtr();
			auto bn2 = shapeFrame2->asShapeNode()->getBodyNodePtr();

			auto skel1 = bn1->getSkeleton();
			auto skel2 = bn2->getSkeleton();

			if(bn1->getName().find("Foot") != std::string::npos)
				continue;
			else if(bn2->getName().find("Foot") != std::string::npos)
				continue;

			if(skel1->getName() == "humanoid" && skel2->getName() == "ground"){
				mContactEOE = true;
				break;
			}

			if(skel1->getName() == "ground" && skel2->getName() == "humanoid"){
				mContactEOE = true;
				break;
			}

		}
	}
	if(mEnableGoal)
	{
		this->recordGoal();
		this->updateGoal();
	}
	
	this->recordState();

	mPrevPositions2 = mPrevPositions;
	mPrevPositions = mSimCharacter->getSkeleton()->getPositions();
	mPrevCOM = mSimCharacter->getSkeleton()->getCOM();

	mElapsedFrame++;
	mFrame++;
}
void
Environment::
resetGoal()
{
	// Eigen::Isometry3d T_ref = mSimCharacter->getReferenceTransform();
	// Eigen::Matrix3d R_ref = T_ref.linear();
	// Eigen::AngleAxisd aa_ref(R_ref);
	// double heading = aa_ref.angle()*aa_ref.axis()[1];
	//Eigen::Vector3d heading = R_ref.inverse() * Eigen::Vector3d::UnitZ();
	// this->mTargetHeading = heading-M_PI/2;

	// mTargetSpeed = dart::math::Random::uniform<double>(mTargetSpeedMin, mTargetSpeedMax);
	// Eigen::Vector3d com_vel = mSimCharacter->getSkeleton()->getCOMLinearVelocity();
	// com_vel[1] =0.0;
	// if(std::abs(com_vel[0])>1e-5) this->mTargetHeading = std::atan(com_vel[2]/com_vel[0]);
	// else{
	// 	this->mTargetHeading = com_vel[2]>0? 90: 270; 
	// }
	// this->mTargetSpeed = std::max(1.0, com_vel.norm());
	// this->mTargetHeight = mSimCharacter->getSkeleton()->getCOM()[1];
	// this->mIdleHeight = mSimCharacter->getSkeleton()->getCOM()[1];

	return;
}
void
Environment::
updateGoal()
{
	// bool sharp_turn = dart::math::Random::uniform<double>(0.0, 1.0)<mSharpTurnProb?true:false;
	// double delta_heading = 0;
	// if(sharp_turn)
	// 	delta_heading = dart::math::Random::uniform<double>(-M_PI, M_PI);
	// else
	// 	delta_heading = dart::math::Random::normal<double>(0.0, mMaxHeadingTurnRate);
	// mTargetHeading += delta_heading;

	// bool change_speed = dart::math::Random::uniform<double>(0.0, 1.0)<mSpeedChangeProb?true:false;
	// if(change_speed)
	// 	mTargetSpeed = dart::math::Random::uniform(mTargetSpeedMin, mTargetSpeedMax);

	// bool change_height = dart::math::Random::uniform<double>(0.0, 1.0)<mHeightChangeProb?true:false;
	// if(change_height)
	// 	mTargetHeight = dart::math::Random::uniform(mTargetHeightMin, mTargetHeightMax);

	bool change_motion = dart::math::Random::uniform<double>(0.0, 1.0)<mTransitionProb?true:false;
	if(change_motion)
		mStateLabel = dart::math::Random::uniform<int>(0, this->mNumMotions-1);


	return;
}
void
Environment::
recordGoal()
{
	mRewardGoal = 1.0;
	// Eigen::Isometry3d T_ref = mSimCharacter->getReferenceTransform();
	// Eigen::Matrix3d R_ref = T_ref.linear();

	// Eigen::Vector3d com_vel = (mSimCharacter->getSkeleton()->getCOM() - mPrevCOM)*mControlHz;
	// com_vel[1] = 0.0;
	// com_vel = R_ref.inverse() * com_vel;

	// // Eigen::Vector3d target_direction = R_target.col(2);
	// Eigen::Vector3d target_direction(std::cos(mTargetHeading), 0.0, -std::sin(mTargetHeading));

	// mTargetDirection = target_direction;
	// Eigen::Vector3d tar_loc = R_ref.inverse() * target_direction;

	// double err_height = mTargetHeight - mSimCharacter->getSkeleton()->getCOM()[1];
	
	// double err_vel = std::sqrt(mTargetHeight *2* 9.8) - com_vel[1];

	// mStateGoal.resize(7+mNumMotions);
	// mStateGoal<<com_vel, tar_loc, 1.0, mStateLabel;
	
	mStateGoal.resize(mDimLabel);
	mStateGoal<<mStateLabel;


	// double proj_vel = tar_loc.dot(com_vel);
	// mRewardGoal = 0.0;
	// if(proj_vel > 0.0)
	// {
	// 	double err = std::max(1.0 - proj_vel, 0.0);
	// 	mRewardGoal = std::exp(-1.0*err*err);
	// }

	// Eigen::Vector3d vel = mSimCharacter->getSkeleton()->getCOMLinearVelocity();
	// double vel_reward=0;
	// if(vel[1]>0){
	// 	double vel_reward = std::exp(-1.0 * err_vel * err_vel);
	// }

	// double pos_reward = 0.5;
	// if(mSimCharacter->getSkeleton()->getCOM()[1] > 0.86){
	// 	pos_reward += 0.5 * std::exp(-10.0 * err_height * err_height);
	// }

	// mRewardGoal *= (0.5 * pos_reward + 0.5 * vel_reward);
}

double
Environment::
getRewardGoal()
{
	return mRewardGoal;
}

int
Environment::
getStateLabel()
{
	return mStateLabel;
}

void
Environment::
setStateLabel(int label)
{
	mStateLabel = label;
	return;
}

const Eigen::VectorXd&
Environment::
getStateGoal()
{
	return mStateGoal;
}

const Eigen::VectorXd&
Environment::
getState()
{
	return mState;
}
const Eigen::VectorXd&
Environment::
getStateAMP()
{
	return mStateAMP;
}
void
Environment::
recordState()
{
	Eigen::VectorXd state = MathUtils::ravel(mSimCharacter->getState());
	if(mEnableGoal)
	{
		Eigen::VectorXd goal = this->getStateGoal();
		mState = Eigen::VectorXd(state.rows() + goal.rows());
		mState<<state, goal;	
	}
	else
		mState = state;	
	
	auto save_state = mKinCharacter->saveState();

	Eigen::VectorXd prev_velocities = mSimCharacter->computeAvgVelocity(mPrevPositions2, mPrevPositions, 1.0/mControlHz);
	mKinCharacter->getSkeleton()->setPositions(mPrevPositions);
	mKinCharacter->getSkeleton()->setVelocities(prev_velocities);
	
	Eigen::VectorXd s = mKinCharacter->getStateAMP();

	Eigen::VectorXd velocities = mSimCharacter->computeAvgVelocity(mPrevPositions, mSimCharacter->getSkeleton()->getPositions(), 1.0/mControlHz);
	mKinCharacter->getSkeleton()->setPositions(mSimCharacter->getSkeleton()->getPositions());
	mKinCharacter->getSkeleton()->setVelocities(velocities);

	Eigen::VectorXd s1 = mKinCharacter->getStateAMP();
	mKinCharacter->restoreState(save_state);
	mStateAMP.resize(s.rows() + s1.rows()+mDimLabel);
	mStateAMP<<s, s1, mStateLabel;
}


Eigen::MatrixXd
Environment::
getStateAMPExpert()
{
	int total_num_frames = 0;
	int m = this->getDimStateAMP();
	int m2 = (m-1)/2;
	int o = 0;
	for(auto motion: mMotions)
	{
		int nf = motion->getNumFrames();
		total_num_frames += nf-1;
	}
	Eigen::MatrixXd state_expert(total_num_frames,m);

	for(int n=0; n<mNumMotions; n++)
	{
		auto motion = mMotions[n];
		int label= n;

		int nf = motion->getNumFrames();
		mKinCharacter->setPose(motion->getPosition(0),
							motion->getRotation(0),
							motion->getLinearVelocity(0),
							motion->getAngularVelocity(0));
		Eigen::VectorXd s = mKinCharacter->getStateAMP();

		Eigen::VectorXd s1;

		for(int i=0;i<nf-1;i++)
		{
			mKinCharacter->setPose(motion->getPosition(i+1),
							motion->getRotation(i+1),
							motion->getLinearVelocity(i+1),
							motion->getAngularVelocity(i+1));
			s1 = mKinCharacter->getStateAMP();

			state_expert.row(o+i).head(m2) = s.transpose();
			state_expert.row(o+i).segment(m2,m2) = s1.transpose();
			state_expert.row(o+i)[m-1] = label;
			s = s1;
		}
		o += nf - 1;
	}
	return state_expert;
}
void
Environment::
FollowBVH(int idx){

	auto& motion = mMotions[idx];
	
	Eigen::Vector3d position = motion->getPosition(mFrame);
	Eigen::MatrixXd rotation = motion->getRotation(mFrame);
	Eigen::Vector3d linear_velocity = motion->getLinearVelocity(mFrame);
	Eigen::MatrixXd angular_velocity = motion->getAngularVelocity(mFrame);
	mKinCharacter->setPose(position, rotation, linear_velocity, angular_velocity);
	if(mFrame > (motion->getNumFrames()-3))
		mFrame = 0;
	return;
}
bool
Environment::
inspectEndOfEpisode()
{
	if(mContactEOE)
		return true;
	else if(mElapsedFrame>mMaxElapsedFrame)
		return true;

	return false;
}

double
Environment::
computeGroundHeight()
{
	double y = 1e6;

	for(auto motion: mMotions)
	{
		int nf = motion->getNumFrames();
		for(int i=0;i<nf;i++)
		{
			mKinCharacter->setPose(motion->getPosition(i),
							motion->getRotation(i),
							motion->getLinearVelocity(i),
							motion->getAngularVelocity(i));

			y= std::min(y, mKinCharacter->getSkeleton()->getBodyNode("LeftFoot")->getCOM()[1]);
			y= std::min(y, mKinCharacter->getSkeleton()->getBodyNode("RightFoot")->getCOM()[1]);
		}
	}
	
	float dy = dynamic_cast<const BoxShape*>(mKinCharacter->getSkeleton()->getBodyNode("LeftFoot")->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1]*0.5;
	return y - dy;
}
Eigen::MatrixXd
Environment::
getActionSpace()
{
	Eigen::MatrixXd action_space = Eigen::MatrixXd::Ones(this->getDimAction(), 2);
	int n = mSimCharacter->getSkeleton()->getNumDofs();

	action_space.col(0) *= -M_PI*2; // Lower
	action_space.col(1) *=  M_PI*2; // Upper

	return action_space;
}
Eigen::VectorXd
Environment::
getActionWeight()
{
	Eigen::VectorXd action_weight = Eigen::VectorXd::Ones(this->getDimAction());
	int n = mSimCharacter->getSkeleton()->getNumDofs();

	return action_weight;
}
Eigen::VectorXd
Environment::
convertToRealActionSpace(const Eigen::VectorXd& a_norm)
{
	Eigen::VectorXd a_real;
	Eigen::VectorXd lo = mActionSpace.col(0), hi =  mActionSpace.col(1);
	a_real = dart::math::clip<Eigen::VectorXd, Eigen::VectorXd>(a_norm, lo, hi);
	a_real = mActionWeight.cwiseProduct(a_real);
	return a_real;
}

double
Environment::
getTargetHeading()
{
	return this->mTargetHeading;
}

double
Environment::
getTargetSpeed()
{
	return this->mTargetSpeed;
}

double
Environment::
getTargetHeight()
{
	return this->mTargetHeight;
}

const Eigen::Vector3d
Environment::
getTargetDirection()
{
	return this->mTargetDirection;
}

void
Environment::
setTargetHeading(double heading)
{
	this->mTargetHeading = heading;
	return;
}

void
Environment::
setTargetSpeed(double speed)
{
	this->mTargetSpeed = speed;
	return;
}
void
Environment::
setTargetHeight(double height)
{
	this->mTargetHeight = height;
	return;
}
