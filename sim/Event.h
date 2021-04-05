#ifndef __EVENT_H__
#define __EVENT_H__
#include "dart/dart.hpp"

class Environment;
class Character;
class Event
{
public:
	Event(Environment* env, Character* chararacter);

	virtual void call() = 0;
	virtual void reset() = 0;
protected:
	Environment* mEnvironment;
	Character* mCharacter;
};

class ObstacleEvent : public Event
{
public:
	ObstacleEvent(Environment* env, Character* chararacter);

	void call() override;
	void reset() override;
	const dart::dynamics::SkeletonPtr& getObstacle(){return mObstacle;}

	const Eigen::Vector3d& getLinearPosition(){return mLinearPosition;}
	const Eigen::Vector3d& getLinearVelocity(){return mLinearVelocity;}

	void setLinearPosition(const Eigen::Vector3d& linear_position){mLinearPosition = linear_position;}
	void setLinearVelocity(const Eigen::Vector3d& linear_velocity){mLinearVelocity = linear_velocity;}
protected:
	dart::dynamics::SkeletonPtr mObstacle;
	int mTriggerCount;
	int mTriggerDuration;

	Eigen::Vector3d mLinearPosition;
	Eigen::Vector3d mLinearVelocity;
};

#endif