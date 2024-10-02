#include "integrator.h"
#include <iostream>

#include <vector>
namespace simulation {
// Factory
std::unique_ptr<Integrator> IntegratorFactory::CreateIntegrator(IntegratorType type) {
    switch (type) {
        case simulation::IntegratorType::ExplicitEuler:
            return std::make_unique<ExplicitEulerIntegrator>();
        case simulation::IntegratorType::ImplicitEuler:
            return std::make_unique<ImplicitEulerIntegrator>();
        case simulation::IntegratorType::MidpointEuler:
            return std::make_unique<MidpointEulerIntegrator>();
        case simulation::IntegratorType::RungeKuttaFourth:
            return std::make_unique<RungeKuttaFourthIntegrator>();
        default:
            throw std::invalid_argument("TerrainFactory::CreateTerrain : invalid TerrainType");
            break;
    }
    return nullptr;
}

//
// ExplicitEulerIntegrator
//

IntegratorType ExplicitEulerIntegrator::getType() { return IntegratorType::ExplicitEuler; }

void ExplicitEulerIntegrator::integrate(MassSpringSystem& particleSystem) {

    // TODO#4-1: Integrate position and velocity
    //   1. Integrate position using current velocity.
    //   2. Integrate velocity using current acceleration.
    //   3. Clear force
    // Note:
    //   1. You should do this first. Then you can check whether your collision is correct or not.
    //   2. See functions in `particle.h` if you don't know how to access data.
    //   3. Review ¡§ODE_basics.pptx¡¨ from p.15 - p.16

    for (int i = 0; i < particleSystem.jellyCount; i++) {
        for (int j = 0; j < particleSystem.jellies[i].getParticleNum(); j++) {
            Particle &p = particleSystem.jellies[i].getParticle(j);
            p.setPosition(p.getPosition() + particleSystem.deltaTime * p.getVelocity());
            p.setVelocity(p.getVelocity() + particleSystem.deltaTime * p.getAcceleration());
            p.setForce(Eigen::Vector3f::Zero());
        }
    }

}

//
// ImplicitEulerIntegrator
//

IntegratorType ImplicitEulerIntegrator::getType() { return IntegratorType::ImplicitEuler; }

void ImplicitEulerIntegrator::integrate(MassSpringSystem& particleSystem) {
    // TODO#4-2: Integrate position and velocity
    //   1. Backup original particles' data.
    //   2. Integrate position and velocity using explicit euler to get Xn+1.
    //   3. Compute refined Xn+1 and Vn+1 using (1.) and (2.).
    // Note:
    //   1. Use `MassSpringSystem::computeJellyForce` with modified position and velocity to get Xn+1
    //   2. Review ¡§ODE_basics.pptx¡¨ from p.18 - p.19

    std::vector<Particle> ori;

    for (int i = 0; i < particleSystem.jellyCount; i++) {
        for (int j = 0; j < particleSystem.jellies[i].getParticleNum(); j++) {
            Particle tp;
            Particle& p = particleSystem.jellies[i].getParticle(j);
            tp.setPosition(p.getPosition());
            tp.setVelocity(p.getVelocity());
            ori.push_back(tp);

            p.setPosition(p.getPosition() + particleSystem.deltaTime * p.getVelocity());
            p.setVelocity(p.getVelocity() + particleSystem.deltaTime * p.getAcceleration());
            p.setForce(Eigen::Vector3f::Zero());
        }
    }
    particleSystem.computeJellyForce(particleSystem.jellies[0]);
    
    for (int i = 0; i < particleSystem.jellyCount; i++) {
        for (int j = 0; j < particleSystem.jellies[i].getParticleNum(); j++) {
            Particle& p = particleSystem.jellies[i].getParticle(j);
            p.setVelocity(ori[j].getVelocity() + particleSystem.deltaTime * p.getAcceleration());
            p.setPosition(ori[j].getPosition() + particleSystem.deltaTime * ori[j].getVelocity());
        }
    }
    
}

//
// MidpointEulerIntegrator
//

IntegratorType MidpointEulerIntegrator::getType() { return IntegratorType::MidpointEuler; }

void MidpointEulerIntegrator::integrate(MassSpringSystem& particleSystem) {
    // TODO#4-3: Integrate position and velocity
    //   1. Backup original particles' data.
    //   2. Integrate position and velocity using explicit euler to get Xn+1.
    //   3. Compute refined Xn+1 using (1.) and (2.).
    // Note:
    //   1. Use `MassSpringSystem::computeJellyForce` with modified position and velocity to get Xn+1.
    //   2. Review ¡§ODE_basics.pptx¡¨ from p .18 - p .20

    std::vector<Particle> ori;

    for (int i = 0; i < particleSystem.jellyCount; i++) {
        for (int j = 0; j < particleSystem.jellies[i].getParticleNum(); j++) {
            Particle tp;
            Particle& p = particleSystem.jellies[i].getParticle(j);
            tp.setPosition(p.getPosition());
            tp.setVelocity(p.getVelocity());
            ori.push_back(tp);

            p.setPosition(p.getPosition() + (particleSystem.deltaTime * p.getVelocity()) / 2);
            p.setVelocity(p.getVelocity() + (particleSystem.deltaTime * p.getAcceleration()) / 2);
            p.setForce(Eigen::Vector3f::Zero());
        }
    }
    particleSystem.computeJellyForce(particleSystem.jellies[0]);

    for (int i = 0; i < particleSystem.jellyCount; i++) {
        for (int j = 0; j < particleSystem.jellies[i].getParticleNum(); j++) {
            Particle& p = particleSystem.jellies[i].getParticle(j);
            p.setVelocity(ori[j].getVelocity() + particleSystem.deltaTime * p.getAcceleration());
            p.setPosition(ori[j].getPosition() + particleSystem.deltaTime * ori[j].getVelocity());
        }
    }

}

//
// RungeKuttaFourthIntegrator
//

IntegratorType RungeKuttaFourthIntegrator::getType() { return IntegratorType::RungeKuttaFourth; }

void RungeKuttaFourthIntegrator::integrate(MassSpringSystem& particleSystem) {
    
    // TODO#4-4: Integrate velocity and acceleration
    //   1. Backup original particles' data.
    //   2. Compute k1, k2, k3, k4
    //   3. Compute refined Xn+1 using (1.) and (2.).
    // Note:
    //   1. Use `MassSpringSystem::computeJellyForce` with modified position and velocity to get Xn+1.
    //   2. StateStep struct is just a hint, you can use whatever you want.
    //   3. Review ¡§ODE_basics.pptx¡¨ from p.21
    struct StateStep {
        Eigen::Vector3f vel;
        Eigen::Vector3f pos;
        Eigen::Vector3f acc;
    };

    std::vector<StateStep> ori, k1, k2, k3, k4;
    //k1
    for (int i = 0; i < particleSystem.jellyCount; i++) {
        for (int j = 0; j < particleSystem.jellies[i].getParticleNum(); j++) {
            StateStep t, tk1;
            Particle& p = particleSystem.jellies[i].getParticle(j);

            t.vel = p.getVelocity();
            t.pos = p.getPosition();
            tk1.acc = particleSystem.deltaTime * p.getAcceleration();
            tk1.vel = particleSystem.deltaTime * p.getVelocity();
            p.setVelocity(p.getVelocity() + particleSystem.deltaTime * p.getAcceleration() / 2);
            p.setPosition( p.getPosition() + particleSystem.deltaTime * p.getVelocity() / 2);

            ori.push_back(t);
            k1.push_back(tk1);
        }
    }
    particleSystem.computeJellyForce(particleSystem.jellies[0]);

    // k2
    for (int i = 0; i < particleSystem.jellyCount; i++) {
        for (int j = 0; j < particleSystem.jellies[i].getParticleNum(); j++) {
            StateStep tk2;
            Particle& p = particleSystem.jellies[i].getParticle(j);

            tk2.acc = particleSystem.deltaTime * p.getAcceleration();
            tk2.vel = particleSystem.deltaTime * p.getVelocity();
            p.setVelocity(ori[j].vel + particleSystem.deltaTime * p.getAcceleration() / 2);
            p.setPosition(ori[j].pos + particleSystem.deltaTime * p.getVelocity() / 2);

            k2.push_back(tk2);
        }
    }
    particleSystem.computeJellyForce(particleSystem.jellies[0]);

    // k3
    for (int i = 0; i < particleSystem.jellyCount; i++) {
        for (int j = 0; j < particleSystem.jellies[i].getParticleNum(); j++) {
            StateStep tk3;
            Particle& p = particleSystem.jellies[i].getParticle(j);

            tk3.acc = particleSystem.deltaTime * p.getAcceleration();
            tk3.vel = particleSystem.deltaTime * p.getVelocity();
            p.setVelocity(ori[j].vel + particleSystem.deltaTime * p.getAcceleration());
            p.setPosition(ori[j].pos + particleSystem.deltaTime * p.getVelocity());

            k3.push_back(tk3);
        }
    }
    particleSystem.computeJellyForce(particleSystem.jellies[0]);

    //k4
    for (int i = 0; i < particleSystem.jellyCount; i++) {
        for (int j = 0; j < particleSystem.jellies[i].getParticleNum(); j++) {
            StateStep tk4;
            Particle& p = particleSystem.jellies[i].getParticle(j);

            tk4.acc = particleSystem.deltaTime * p.getAcceleration();
            tk4.vel = particleSystem.deltaTime * p.getVelocity();
            p.setVelocity(ori[j].vel + (k1[j].acc + 2 * k2[j].acc + 2 * k3[j].acc + tk4.acc) / 6);
            p.setPosition(ori[j].pos + (k1[j].vel + 2 * k2[j].vel + 2 * k3[j].vel + tk4.vel) / 6);

            k4.push_back(tk4);
        }
    }
}
}  // namespace simulation
