#include "massSpringSystem.h"

#include <cmath>
#include <iostream>
#include <utility>

#include "integrator.h"
namespace simulation {
constexpr float g_cdDeltaT = 0.001f;
constexpr float g_cdK = 2000.0f;
constexpr float g_cdD = 60.0f;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Constructor & Destructor
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

MassSpringSystem::MassSpringSystem()
    : 
      isDrawingJelly(true),
      isDrawingStruct(false),
      isDrawingShear(false),
      isDrawingBending(false),
      isSimulating(false),

      integratorType(IntegratorType::ExplicitEuler),
      particleCountPerEdge(10),
      jellyCount(1),
      jellyID(1),

      deltaTime(g_cdDeltaT),
      springCoefStruct(g_cdK),
      springCoefShear(g_cdK),
      springCoefBending(g_cdK),
      damperCoefStruct(g_cdD),
      damperCoefShear(g_cdD),
      damperCoefBending(g_cdD),
      rotation(0.0),
      jellyLength(4),

      position(Eigen::Vector3f(-4.8f, 7.0f, 0.0f)),
      gravity(Eigen::Vector3f(0.0f, -9.8f, 0.0f)) {
    initializeJelly();
    reset();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Set and Update
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void MassSpringSystem::reset() {

    for (int jellyIdx = 0; jellyIdx < jellyCount; jellyIdx++) {
        jellies[jellyIdx].resetJelly(position, rotation);
    }


}

void MassSpringSystem::setSpringCoef(const float springCoef, const Spring::SpringType springType) {


    for (int jellyIdx = 0; jellyIdx < jellyCount; jellyIdx++) {
        if (springType == Spring::SpringType::STRUCT) {
            springCoefStruct = springCoef;
            jellies[jellyIdx].setSpringCoef(springCoef, Spring::SpringType::STRUCT);
        } else if (springType == Spring::SpringType::SHEAR) {
            springCoefShear = springCoef;
            jellies[jellyIdx].setSpringCoef(springCoef, Spring::SpringType::SHEAR);
        } else if (springType == Spring::SpringType::BENDING) {
            springCoefBending = springCoef;
            jellies[jellyIdx].setSpringCoef(springCoef, Spring::SpringType::BENDING);
        } else {
            std::cout << "Error spring type in MassSpringSystem SetSpringCoef" << std::endl;
        }
    }
}

void MassSpringSystem::setDamperCoef(const float damperCoef, const Spring::SpringType springType) {



    for (int jellyIdx = 0; jellyIdx < jellyCount; jellyIdx++) {
        if (springType == Spring::SpringType::STRUCT) {
            damperCoefStruct = damperCoef;
            jellies[jellyIdx].setDamperCoef(damperCoef, Spring::SpringType::STRUCT);
        } else if (springType == Spring::SpringType::SHEAR) {
            damperCoefShear = damperCoef;
            jellies[jellyIdx].setDamperCoef(damperCoef, Spring::SpringType::SHEAR);
        } else if (springType == Spring::SpringType::BENDING) {
            damperCoefBending = damperCoef;
            jellies[jellyIdx].setDamperCoef(damperCoef, Spring::SpringType::BENDING);
        } else {
            std::cout << "Error spring type in CMassSpringSystme SetDamperCoef" << std::endl;
        }
    }



}

void MassSpringSystem::setTerrain(std::unique_ptr<Terrain>&& terrain, std::string planeType) {
    if (planeType == "BOWL") {
        this->bowlTerrain = std::move(terrain);
    }
    if (planeType == "PLANE") {
        this->planeTerrain = std::move(terrain);
    }

}

void MassSpringSystem::setIntegrator(std::unique_ptr<Integrator>&& integrator) {
    this->integrator = std::move(integrator);
    this->integratorType = this->integrator->getType();
}

float MassSpringSystem::getSpringCoef(const Spring::SpringType springType) {
    if (springType == Spring::SpringType::STRUCT) {
        return springCoefStruct;
    } else if (springType == Spring::SpringType::SHEAR) {
        return springCoefShear;
    } else if (springType == Spring::SpringType::BENDING) {
        return springCoefBending;
    } else {
        std::cout << "Error spring type in CMassSpringSystme GetSpringCoef" << std::endl;
        throw std::invalid_argument("MassSpringSystem::GetSpringCoef : invalid springType");
    }
}
float MassSpringSystem::getDamperCoef(const Spring::SpringType springType) {
    if (springType == Spring::SpringType::STRUCT) {
        return damperCoefStruct;
    } else if (springType == Spring::SpringType::SHEAR) {
        return damperCoefShear;
    } else if (springType == Spring::SpringType::BENDING) {
        return damperCoefBending;
    } else {
        std::cout << "Error spring type in CMassSpringSystme GetDamperCoef" << std::endl;
        throw std::invalid_argument("MassSpringSystem::GetDamperCoef : invalid springType");
    }
}

int MassSpringSystem::getJellyCount() const { return jellyCount; }
Jelly* MassSpringSystem::getJellyPointer(int n) {
    if (n >= jellyCount) {
        throw std::runtime_error("Jelly index out of range");
    }
    return &jellies[n];
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Simulation Part
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool MassSpringSystem::checkStable() {
    for (int jellyIdx = 0; jellyIdx < jellyCount; jellyIdx++) {
        Jelly testJelly = jellies[jellyIdx];
        int springNum = testJelly.getSpringNum();

        for (int springIdx = 0; springIdx < springNum; springIdx++) {
            Spring spring = testJelly.getSpring(springIdx);
            float vel = testJelly.getParticle(spring.getSpringStartID()).getVelocity().squaredNorm();

            if (std::isnan(vel) || vel > 1e6) return false;
        }
    }
    return true;
}

void MassSpringSystem::simulationOneTimeStep() {
    // std::cout << "deltaTime : " << this->deltaTime << std::endl;
    if (isSimulating) {
        integrate();
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Initialization
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void MassSpringSystem::initializeJelly() {
    for (int jellyIdx = 0; jellyIdx < jellyCount; jellyIdx++) {
        Jelly NewJelly(Eigen::Vector3f(0.0f, jellyLength, 0.0f - (2.0f * jellyLength * jellyIdx)), jellyLength,
                     particleCountPerEdge, springCoefStruct, damperCoefStruct);
        jellies.push_back(NewJelly);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute Force
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void MassSpringSystem::computeAllForce() {

    for (int jellyIdx = 0; jellyIdx < jellyCount; jellyIdx++) {
        computeJellyForce(jellies[jellyIdx]);
    }
}

void MassSpringSystem::computeJellyForce(Jelly& jelly) {
    jelly.addForceField(gravity);
    jelly.computeInternalForce();
    // delegate to terrain to handle collision
    
    bowlTerrain->handleCollision(deltaTime, jelly);
    planeTerrain->handleCollision(deltaTime, jelly);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Integrator
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void MassSpringSystem::integrate() {
    computeAllForce();
    integrator->integrate(*this);
}
}  // namespace simulation
