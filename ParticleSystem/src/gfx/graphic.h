#pragma once
#include <array>
#include <memory>
#include <vector>

#include "Eigen/Dense"
#include "../simulation/jelly.h"
#include "shader.h"
#include "texture.h"

namespace gfx {
class Rigidbody {
 public:
    Rigidbody();
    Rigidbody(const Rigidbody&) = delete;
    Rigidbody& operator=(const Rigidbody&) = delete;

    void setModelMatrix(const Eigen::Matrix4f& _modelMatrix);
    void setTexture(const std::shared_ptr<Texture>& texture);
    void setTexture(const Eigen::Vector3f& color);

    void initialize();

    virtual void generateVertices() = 0;
    virtual void render(Program* shaderProgram) = 0;

 protected:
    virtual ~Rigidbody();
    GLuint vao;
    GLuint vbo;

    Eigen::Matrix4f modelMatrix = Eigen::Matrix4f::Identity();
    Eigen::Matrix3f inverseTransposeModel = Eigen::Matrix3f::Identity();
    Eigen::Vector3f baseColor = Eigen::Vector3f(1.0, 0.0, 0.0);
    std::shared_ptr<Texture> texture = nullptr;
    std::vector<float>* vertexPointer = nullptr;
};

class Plane final : public Rigidbody {
 public:
    Plane();
    void generateVertices() override;
    void render(Program* shaderProgram) override;

 private:
    static std::vector<GLfloat> vertices;
};

class Bowl final : public Rigidbody {
 public:
    enum class RenderMode { WIREFRAME, FILLED };
    Bowl();
    ~Bowl();
    void setRenderMode(RenderMode _mode);
    void generateVertices() override;
    void render(Program* shaderProgram) override;

 private:
    GLuint wireVAO;
    RenderMode mode = RenderMode::FILLED;
    static std::vector<GLuint> fillIndices;
    static std::vector<GLuint> wireIndices;
    GLuint fillEBO, wireEBO;

 private:
    static std::vector<GLfloat> vertices;
};

class SoftJelly final {
 private:
    const Eigen::Matrix4f modelMatrix = Eigen::Matrix4f::Identity();
    const Eigen::Matrix3f inverseTransposeModel = Eigen::Matrix3f::Identity();
    GLuint jellyVAO[6], structVAO, shearVAO, bendingVAO, particleVAO;
    GLuint jellyVBO[6], vertexVBO, normalVBO[6], textureCoordVBO;
    GLuint jellyEBO[2], structEBO, shearEBO, bendingEBO;

    simulation::Jelly* jelly;

    std::array<std::shared_ptr<Texture>, 6> textures;
    std::vector<GLfloat> fullVertices;
    std::vector<GLfloat> normalBuffer;
    std::vector<GLfloat> faceVertices;
    std::vector<GLuint> structIndices;
    std::vector<GLuint> shearIndices;
    std::vector<GLuint> bendingIndices;
    Eigen::Vector3f baseColor = Eigen::Vector3f(1.0f, 0.0f, 0.0f);

    void calculateIndices();
    void calculateTextureCoords();
    void initializeBuffers();
    void allocateBuffers();
    void updateSingleFace(int face);

 public:
    explicit SoftJelly(simulation::Jelly* _jelly);
    SoftJelly(const SoftJelly&) = delete;
    SoftJelly(SoftJelly&&) = delete;
    SoftJelly& operator=(const SoftJelly&) = delete;
    ~SoftJelly();

    void setTextures(const std::array<std::shared_ptr<Texture>, 6>& _textures);

    void update();

    void renderJelly(Program* shaderProgram);
    void renderPoints(Program* shaderProgram);
    void renderLines(Program* shaderProgram, simulation::Spring::SpringType springType);
};
class SkyBox {
 public:
    SkyBox();
    SkyBox(const SkyBox&) = delete;
    SkyBox& operator=(const SkyBox&) = delete;
    ~SkyBox();

    void setTexture(const std::shared_ptr<CubeTexture>& _texture);
    void render(Program* shaderProgram);

 private:
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;

    std::shared_ptr<CubeTexture> texture = nullptr;
    static std::vector<GLfloat> vertices;
    static std::vector<GLuint> indices;
};
}  // namespace gfx
