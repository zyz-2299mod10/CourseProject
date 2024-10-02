#pragma once
#include <memory>
#include <string>
#include <vector>

#include "Eigen/Dense"

#include "jelly.h"
#include "terrain.h"

namespace simulation {
// forward declaration to avoid circular include
// class Integrator depends on MassSpringSystem, and vice versa.
// so we need to forward declaring them to break the circular include.
class Integrator;
enum class IntegratorType : char;

class MassSpringSystem {
 public:
    bool isDrawingJelly;   // visibility of the jelly and springs
    bool isDrawingStruct;  // struct stands for structural
    bool isDrawingShear;
    bool isDrawingBending;
    bool isSimulating;  // start or pause

    IntegratorType integratorType;
    int jellyCount;  // number of jellies

    int particleCountPerEdge;  // number of particles at jelly's edge
    int jellyID;               // ID of the jelly under control (offset of rotation)

    float deltaTime;  // deltaTime
    float springCoefStruct;
    float springCoefShear;
    float springCoefBending;
    float damperCoefStruct;
    float damperCoefShear;
    float damperCoefBending;

    float rotation;  // rotation around axis (1, 0, 1)
    float jellyLength;

    Eigen::Vector3f position;
    Eigen::Vector3f gravity;  // external force field

    std::vector<Jelly> jellies;

 public:
    MassSpringSystem();

    //==========================================
    //  getter
    //==========================================

    float getSpringCoef(const Spring::SpringType springType);
    float getDamperCoef(const Spring::SpringType springType);
    int getJellyCount() const;
    Jelly* getJellyPointer(int n);
    //==========================================
    //  setter
    //==========================================

    void setSpringCoef(const float springCoef, const Spring::SpringType springType);
    void setDamperCoef(const float damperCoef, const Spring::SpringType springType);

    void setTerrain(std::unique_ptr<Terrain>&& terrain, std::string planeTSype);
    void setIntegrator(std::unique_ptr<Integrator>&& integrator);

    //==========================================
    //  method
    //==========================================

    bool checkStable();

    void simulationOneTimeStep();

    void reset();

 private:
    // used for delegating part of the simulation process to other classes, in
    // this case : collision detection to terrain, and integration to
    // integrator.
    std::shared_ptr<Terrain> bowlTerrain;
    std::shared_ptr<Terrain> planeTerrain;
    std::vector<std::shared_ptr<Terrain>> terrains;
    std::unique_ptr<Integrator> integrator;

    //==========================================
    //  internal method
    //==========================================

    void initializeJelly();

    void computeAllForce();                     // compute force of whole systems
    void computeJellyForce(Jelly& jelly);  // compute force of one jelly


    void integrate();

    // allow integrators to access internal methods
    friend class ExplicitEulerIntegrator;
    friend class ImplicitEulerIntegrator;
    friend class MidpointEulerIntegrator;
    friend class RungeKuttaFourthIntegrator;
};
}  // namespace simulation
