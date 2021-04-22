#include "DARTRendering.h"
#include "DrawUtils.h"
using namespace dart;
using namespace dart::dynamics;

DARTRendering::Option DARTRendering::gRenderOption = DARTRendering::Option();
void
DARTRendering::
drawLinks(const SkeletonPtr& skel)
{

}
void
DARTRendering::
drawJoints(const SkeletonPtr& skel)
{

}
// void
// DARTRendering::
// drawForceSensors(Character* character, const Eigen::Vector3d& pos, const Eigen::Vector3d& size ,const Option& option)
// {
// 	Eigen::VectorXd rest_pose = Eigen::VectorXd::Zero(character->getSkeleton()->getNumDofs());
// 	int idx = character->getSkeleton()->getJoint("LeftArm")->getIndexInSkeleton(2);
// 	rest_pose[idx] = - 1.4;
// 	idx = character->getSkeleton()->getJoint("RightArm")->getIndexInSkeleton(2);
// 	rest_pose[idx] = + 1.4;
// 	glDisable(GL_LIGHTING);

// 	glMatrixMode(GL_PROJECTION);

// 	glPushMatrix();
// 	glLoadIdentity();

// 	double x_min,x_max, y_min,y_max;
// 	// x_min=-0.3, x_max=0.3, y_min=0.8, y_max=2.0;
// 	x_min=-0.3, x_max=0.3, y_min=-0.2, y_max=1.0;
// 	gluOrtho2D(x_min,x_max, y_min,y_max);

// 	GLint w = glutGet(GLUT_WINDOW_WIDTH);
// 	GLint h = glutGet(GLUT_WINDOW_HEIGHT);
// 	glViewport(w*(pos[0]),h*(pos[1]),w*size[0],h*size[1]);
// 	glMatrixMode(GL_MODELVIEW);
// 	glPushMatrix();
// 	glLoadIdentity();

// 	glColor3f(1,1,1);
// 	glBegin(GL_QUADS);
// 	glVertex3f(x_min,y_min,0.0);
// 	glVertex3f(x_min,y_max,0.0);
// 	glVertex3f(x_max,y_max,0.0);
// 	glVertex3f(x_max,y_min,0.0);
// 	glEnd();
	
// 	Eigen::VectorXd old_pos = character->getSkeleton()->getPositions();
// 	Option option2 = option;
// 	option2.draw_mode = DrawUtils::eDrawWireSimple;
// 	option2.drawJoints = false;
// 	option2.line_width =1.5;
// 	glDisable(GL_DEPTH_TEST);
// 	character->getSkeleton()->setPositions(rest_pose);
// 	drawSkeleton(character->getSkeleton(), option2);

// 	Eigen::VectorXd applied_forces = character->getAppliedForces();
// 	for(int i=1;i<character->getSkeleton()->getNumJoints();i++)
// 	{
// 		auto joint =  character->getSkeleton()->getJoint(i); 
// 		auto parent = joint->getParentBodyNode();

// 		Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
// 		T = parent->getTransform()*joint->getTransformFromParentBodyNode();

// 		double radius = 0.02;
// 		Eigen::Vector3d applied_force = Eigen::Vector3d::Zero();
// 		if(joint->getType()=="BallJoint")
// 			 applied_force = applied_forces.segment<3>(joint->getIndexInSkeleton(0));

// 		double mag = 0.05*applied_force.norm();
// 		glPushMatrix();
// 		glMultMatrixd(T.data());

// 		glDisable(GL_LIGHTING);
// 		glDisable(GL_TEXTURE_2D);
// 		glColor3f(0.4+0.6*mag,0.2,0.2);
// 		DrawUtils::drawSphere(radius*std::sqrt(1.0+mag));
// 		glEnable(GL_TEXTURE_2D);
// 		glEnable(GL_LIGHTING);
// 		glPopMatrix();
// 	}
// 	auto force_sensors = character->getForceSensors();
// 	double sensor_radius = option.sensor_radius;

// 	for(auto fs : force_sensors)
// 	{
// 		glPushMatrix();
// 		Eigen::Vector3d p = fs->getPosition();
// 		Eigen::Vector3d p_haptic = fs->getBodyNode()->getTransform().linear()*fs->getHapticPosition();
// 		// std::cout<<p_haptic.transpose()<<std::endl;
// 		DrawUtils::translate(p+p_haptic);
		
// 		// if(fs->value.norm()>1e-6){
// 		// 	// sensor_radius = option.sensor_radius*std::sqrt(fs->value.norm());
// 		// 	// sensor_radius = option.sensor_radius*std::sqrt(fs->value.norm())*0.5;
// 		// 	glColor3f(1.0,0.1,0.0);
// 		// }
// 		// else{
// 		// 	// sensor_radius = option.sensor_radius;
// 		// 	glColor3f(0.6,0.25,0.0);
// 		// }
// 		glColor3f(0.0,0.0,0.0);
// 		glDisable(GL_LIGHTING);
// 		glDisable(GL_TEXTURE_2D);
// 		DrawUtils::drawSphere(sensor_radius);
// 		DrawUtils::drawSphere(sensor_radius*1.05,DrawUtils::eDrawWireSimple);
// 		glEnable(GL_TEXTURE_2D);
// 		glEnable(GL_LIGHTING);
// 		glPopMatrix();
    	
// 	}
// 	character->getSkeleton()->setPositions(old_pos);
// 	glEnable(GL_DEPTH_TEST);
// 	glPopMatrix();
// 	glMatrixMode(GL_PROJECTION);
// 	glPopMatrix();
// 	glMatrixMode(GL_MODELVIEW);

// 	glEnable(GL_LIGHTING);
// 	glViewport(0,0,w,h);
// }

void
DARTRendering::
drawSkeleton(const SkeletonPtr& skel,const Option& option)
{
	if(gRenderOption.texture_id == 0)
		gRenderOption.texture_id = MeshUtils::buildTexture((std::string(ROOT_DIR)+"/data/object.png").c_str());
	DrawUtils::enableTexture(option.texture_id);
	
	glLineWidth(option.line_width);
	if(option.drawLinks)
	for(int i=0;i<skel->getNumBodyNodes();i++)
	{
		auto bn = skel->getBodyNode(i);
		auto shapeNodes = bn->getShapeNodesWith<VisualAspect>();

		auto T = shapeNodes.back()->getTransform();
		// if(T.translation()[1]>1.0)
			
		drawShape(T,shapeNodes.back()->getShape().get(), option.draw_mode);
	}
	// if(option.drawJoints)
	for(int i =0;i<skel->getNumJoints();i++)
	{
		auto parent = skel->getJoint(i)->getParentBodyNode();
		auto child = skel->getJoint(i)->getChildBodyNode();
		auto shapeNodes = child->getShapeNodesWith<VisualAspect>();
		auto shape = shapeNodes.back()->getShape().get();
		double volume = std::cbrt(shape->getVolume())*0.3;
		if(skel->getJoint(i)->getType()=="FreeJoint")
			continue;
		else if(skel->getJoint(i)->getType()=="BallJoint")
			glColor3f(0.8,0.2,0.2);
		else if(skel->getJoint(i)->getType()=="RevoluteJoint")
			glColor3f(0.2,0.8,0.2);
		Eigen::Isometry3d T;
		T.setIdentity();
		if(parent!=nullptr)
			T = parent->getTransform();

		T = T*skel->getJoint(i)->getTransformFromParentBodyNode();
		glPushMatrix();
		glMultMatrixd(T.data());
		glColor3f(0.8,0.8,0.8);
		glDisable(GL_LIGHTING);
		glDisable(GL_TEXTURE_2D);
		if(option.draw_mode != DrawUtils::eDrawWireSimple)
			DrawUtils::drawSphere(volume, option.draw_mode);
		glColor3f(0.0,0.0,0.0);
		DrawUtils::drawSphere(volume*1.05,DrawUtils::eDrawWireSimple);
		glEnable(GL_TEXTURE_2D);
		glEnable(GL_LIGHTING);
		glPopMatrix();
    	
	}
	DrawUtils::disableTexture();
}

void
DARTRendering::
drawShape(const Eigen::Isometry3d& T,
	const dart::dynamics::Shape* shape,
	DrawUtils::eDrawMode draw_mode)
{
	glEnable(GL_LIGHTING);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	glPushMatrix();
	glMultMatrixd(T.data());
	if(shape->is<SphereShape>())
	{
		const auto* sphere = dynamic_cast<const SphereShape*>(shape);
		glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
		
		if(draw_mode != DrawUtils::eDrawWireSimple)
			DrawUtils::drawSphere(sphere->getRadius(), draw_mode);
		glDisable(GL_LIGHTING);
		glDisable(GL_TEXTURE_2D);
		glColor3f(0.0,0.0,0.0);
		DrawUtils::drawSphere(sphere->getRadius()*1.01,DrawUtils::eDrawWireSimple);
		glEnable(GL_TEXTURE_2D);
		glEnable(GL_LIGHTING);
	}
	else if (shape->is<BoxShape>())
	{
		const auto* box = dynamic_cast<const BoxShape*>(shape);
		glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
    	DrawUtils::drawBox(Eigen::Vector3d::Zero(), box->getSize(), draw_mode);
	}

	glPopMatrix();
}