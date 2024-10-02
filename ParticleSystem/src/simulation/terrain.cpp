#include "terrain.h"

#include <stdexcept>

#include "../util/helper.h"
#include <iostream>

namespace simulation {
// Factory
std::unique_ptr<Terrain> TerrainFactory::CreateTerrain(TerrainType type) {
    switch (type) {
        case simulation::TerrainType::Plane:
            return std::make_unique<PlaneTerrain>();
        case simulation::TerrainType::Bowl:
            return std::make_unique<BowlTerrain>();
        default:
            throw std::invalid_argument("TerrainFactory::CreateTerrain : invalid TerrainType");
            break;
    }
    return nullptr;
}
// Terrain

Eigen::Matrix4f Terrain::getModelMatrix() { return modelMatrix; }

// Note:
// You should update each particles' velocity (base on the equation in
// slide) and force (contact force : resist + friction) in handleCollision function

// PlaneTerrain //

PlaneTerrain::PlaneTerrain() { modelMatrix = util::translate(0.0f, position[1], 0.0f) * util::scale(60, 1, 60); }

TerrainType PlaneTerrain::getType() { return TerrainType::Plane; }

void PlaneTerrain::handleCollision(const float delta_T, Jelly& jelly) {
    constexpr float eEPSILON = 0.01f;
    constexpr float coefResist = 0.8f;
    constexpr float coefFriction = 0.2f;
    // TODO#3-1: Handle collision when a particle collide with the plane terrain.
    //   If collision happens:
    //      1. Directly update particles' velocity
    //      2. Apply contact force to particles when needed
    // Note:
    //   1. There are `jelly.getParticleNum()` particles.
    //   2. See TODOs in `Jelly::computeInternalForce` and functions in `particle.h` if you don't know how to access
    //   data.
    // Hint:
    //   1. Review "particles.pptx¡¨ from p.14 - p.19
    //   1. Use a.norm() to get length of a.
    //   2. Use a.normalize() to normalize a inplace.
    //          a.normalized() will create a new vector.
    //   3. Use a.dot(b) to get dot product of a and b.
    
    for (int i = 0; i < jelly.getParticleNum(); i++) {
        Particle &p = jelly.getParticle(i);
        Eigen::Vector3f pp = p.getPosition();
        // collision detect
        float ifonbowl = (pp - hole_position).norm();
        if (ifonbowl <= hole_radius) continue;

        float distance = (pp - position).dot(normal);
        if (distance < eEPSILON && p.getVelocity().dot(normal) < eEPSILON) {
            // velocity
            Eigen::Vector3f norV = p.getVelocity().dot(normal) * normal;
            Eigen::Vector3f tanV = p.getVelocity() - norV;
            Eigen::Vector3f newV = coefResist * norV + tanV;

            p.setVelocity(newV);

            // force
            Eigen::Vector3f conF = Eigen::Vector3f::Zero();
            Eigen::Vector3f friF = Eigen::Vector3f::Zero();
            Eigen::Vector3f totF;

            if (p.getForce().dot(normal) < 0) {
                conF = -(normal.dot(p.getForce())) * normal;
                friF = -coefFriction * (-normal.dot(p.getForce())) * tanV;
            }
            totF = conF + friF;
            p.addForce(totF);
        }

    }
    

}
// BowlTerrain //

BowlTerrain::BowlTerrain() {
    modelMatrix = util::translate(position) * util::rotateDegree(-90, 0, 0) * util::scale(radius, radius, radius);
}

TerrainType BowlTerrain::getType() { return TerrainType::Bowl; }

void BowlTerrain::handleCollision(const float delta_T, Jelly& jelly) {
    constexpr float eEPSILON = 0.01f;
    constexpr float coefResist = 0.8f;
    constexpr float coefFriction = 0.2f;
    // TODO#3-2: Handle collision when a particle collide with the sphere terrain.
    //   If collision happens:
    //      1. Directly update particles' velocity
    //      2. Apply contact force to particles when needed
    // Note:
    //   1. There are `jelly.getParticleNum()` particles.
    //   2. See TODOs in `Jelly::computeInternalForce` and functions in `particle.h` if you don't know how to access
    //   data.
    // Hint:
    //   1. Review "particles.pptx¡¨ from p.14 - p.19
    //   1. Use a.norm() to get length of a.
    //   2. Use a.normalize() to normalize a inplace.
    //          a.normalized() will create a new vector.
    //   3. Use a.dot(b) to get dot product of a and b.

    for (int i = 0; i < jelly.getParticleNum(); i++) {
        Particle& p = jelly.getParticle(i);
        Eigen::Vector3f pp = p.getPosition();
        float pmass = p.getMass();

        Eigen::Vector3f take_y_vec = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
        float touch_ballwall = (pp - position).norm();
        Eigen::Vector3f normal = (position - pp).normalized();
        float underground = pp.dot(take_y_vec);

        if (underground <= 0 && abs(RADIUS - touch_ballwall) < eEPSILON && normal.dot(p.getVelocity()) < 0) {
            // velocity
            Eigen::Vector3f norV = p.getVelocity().dot(normal) * normal;
            Eigen::Vector3f tanV = p.getVelocity() - norV;
            Eigen::Vector3f newV = coefResist * (norV * (pmass - mass) / (pmass + mass)) + tanV;

            p.setVelocity(newV);

            // force
            Eigen::Vector3f conF = Eigen::Vector3f::Zero();
            Eigen::Vector3f friF = Eigen::Vector3f::Zero();
            Eigen::Vector3f totF;
            
            if (p.getForce().dot(normal) < 0) {
                conF = -(normal.dot(p.getForce())) * normal;
                friF = -coefFriction * (-normal.dot(p.getForce())) * tanV;
            }
            totF = conF + friF;

            p.addForce(totF);
        }
    }
}
}  // namespace simulation
