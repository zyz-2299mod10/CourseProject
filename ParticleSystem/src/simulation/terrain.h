#pragma once
#include <memory>

#include "Eigen/Dense"

#include "jelly.h"
#define RADIUS 9.0
#define HOLE_X 2.0
#define HOLE_Z 1.0

namespace simulation {
class Terrain;

enum class TerrainType : char { Plane, Bowl};

class TerrainFactory final {
 public:
    // no instance, only static usage
    TerrainFactory() = delete;
    static std::unique_ptr<Terrain> CreateTerrain(TerrainType type);
};

// a virtual class
class Terrain {
 public:
    virtual ~Terrain() = default;
    Eigen::Matrix4f getModelMatrix();

    virtual TerrainType getType() = 0;
    virtual void handleCollision(const float delta_T, Jelly& jelly) = 0;

 protected:
    Eigen::Matrix4f modelMatrix = Eigen::Matrix4f::Identity();
};

class PlaneTerrain final : public Terrain {
 public:
    friend class TerrainFactory;

    PlaneTerrain();

    TerrainType getType() override;
    void handleCollision(const float delta_T, Jelly& jelly) override;

 private:
    Eigen::Vector3f position = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
    Eigen::Vector3f normal = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
    Eigen::Vector3f hole_position = Eigen::Vector3f(HOLE_X, position[1], HOLE_Z);
    float hole_radius = RADIUS/sqrt(2);

};

class BowlTerrain final : public Terrain {
 public:
    friend class TerrainFactory;

    BowlTerrain();

    TerrainType getType() override;
    void handleCollision(const float delta_T, Jelly& jelly) override;

 private:
    float radius = RADIUS;
    float mass = 10.0f;
    Eigen::Vector3f position = Eigen::Vector3f(HOLE_X, radius / sqrt(2), HOLE_Z);
};

}  // namespace simulation
