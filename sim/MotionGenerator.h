#ifndef __MOTION_GENERATOR_H__
#define __MOTION_GENERATOR_H__

class Character;
class Motion;
class MassSpringDamperSystem;
class MotionGenerator
{
public:
	MotionGenerator(Character* character, Motion* base_motion);


private:
	Motion* mBaseMotion;
	MassSpringDamperSystem* mMSDSystem;
};

#endif