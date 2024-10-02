#include "simulation/kinematics.h"

#include "Eigen/Dense"
#include <iostream>
#include "acclaim/bone.h"
#include "util/helper.h"

#include <algorithm>
#include <stack>
#include <map>
#define pi 3.14159265358979323846

namespace kinematics {
void forwardSolver(const acclaim::Posture& posture, acclaim::Bone* bone) {
    // TODO#1 (FK)
    // You should set these variables:
    //     bone->start_position = Eigen::Vector4d::Zero();
    //     bone->end_position = Eigen::Vector4d::Zero();
    //     bone->rotation = Eigen::Matrix4d::Zero();
    // The sample above just set everything to zero
    // Hint:
    //   1. posture.bone_translations, posture.bone_rotations
    // Note:
    //   1. This function will be called with bone == root bone of the skeleton
    //   2. we use 4D vector to represent 3D vector, so keep the last dimension as "0"
    //   3. util::rotate{Degree | Radian} {XYZ | ZYX}
    //      e.g. rotateDegreeXYZ(x, y, z) means:
    //      x, y and z are presented in degree rotate z degrees along z - axis first, then y degrees along y - axis, and then x degrees along x - axis 

    bone->start_position = Eigen::Vector4d::Zero();
    bone->end_position = Eigen::Vector4d::Zero();
    bone->rotation = Eigen::Matrix4d::Zero();

    std::map<int, bool> vis;
    std::stack<acclaim::Bone*> q;
    vis[bone->idx] = true;
    bone->start_position = posture.bone_translations[bone->idx];
    bone->end_position = posture.bone_translations[bone->idx];
    q.push(bone);

    while (!q.empty()) {
        acclaim::Bone* temp = q.top();
        q.pop();
        if (temp->idx != 0) {
            temp->start_position = temp->parent->end_position;
            Eigen::Quaternion rota =
                util::rotateDegreeZYX(posture.bone_rotations[temp->idx].x(),
                                      posture.bone_rotations[temp->idx].y(),
                                      posture.bone_rotations[temp->idx].z());
            
            Eigen::Affine3d R = temp->rot_parent_current * rota;
            
            for (acclaim::Bone* i = temp->parent; i != nullptr; i = i->parent) {
                Eigen::Quaternion i_rota =
                    util::rotateDegreeZYX(posture.bone_rotations[i->idx].x(), posture.bone_rotations[i->idx].y(),
                                          posture.bone_rotations[i->idx].z());
                
                R = i->rot_parent_current * i_rota * R;
            }         

            Eigen::Vector4d dir = temp->dir * temp->length;
            temp->end_position = R * dir + temp->start_position;
            temp->rotation = R;

        }

        else {
            temp->rotation = Eigen::Affine3d::Identity();
            temp->start_position = posture.bone_translations[temp->idx];
            temp->end_position = temp->start_position;
        }

        for (acclaim::Bone* i = temp->child; i != nullptr; i = i->sibling) {
            if (vis.find(i->idx) == vis.end()) {
                vis[i->idx] = true;
                q.push(i);
            }
        }
    }

}

std::vector<acclaim::Posture> timeWarper(const std::vector<acclaim::Posture>& postures, int allframe_old, int allframe_new) {

    int total_frames = static_cast<int>(postures.size());
    int total_bones = static_cast<int>(postures[0].bone_rotations.size());
    std::vector<acclaim::Posture> new_postures;

    for (int i = 0; i <= allframe_new; ++i) {
        acclaim::Posture new_poseture(total_bones);
        for (int j = 0; j < total_bones; ++j) {
            // TODO#2 (Time warping)
            // original: |--------------|
            // new     : |----------------------|
            // OR
            // original: |--------------|
            // new     : |-------|
            // You should set these variables:
            //     new_postures[i].bone_translations[j] = Eigen::Vector4d::Zero();
            //     new_postures[i].bone_rotations[j] = Eigen::Vector4d::Zero();
            // The sample above just set everything to zero
            // Hint:
            //   1. Scale the frames.
            //   2. You can use linear interpolation with translations.
            //   3. You should use spherical linear interpolation for rotations.

            new_poseture.bone_translations[j] = Eigen::Vector4d::Zero();
            new_poseture.bone_rotations[j] = Eigen::Vector4d::Zero();

            int low = floor(i * (double)allframe_old / (double)allframe_new);
            int high = ceil((i * (double)allframe_old / (double)allframe_new));
            double r = (i * (double)allframe_old / (double)allframe_new) - low;

            if (i == allframe_new - 1) {
                new_poseture.bone_rotations[j] = postures[allframe_old - 1].bone_rotations[j];
                new_poseture.bone_translations[j] = postures[allframe_old - 1].bone_translations[j];
            }

            else if (low == high) {
                new_poseture.bone_rotations[j] = postures[low].bone_rotations[j];
                new_poseture.bone_translations[j] = postures[low].bone_translations[j];
            }

            else {
                // translation
                new_poseture.bone_translations[j] =
                    r * postures[low].bone_translations[j] + ((double)1 - r) * postures[high].bone_translations[j];

                // rotation
                Eigen::Quaternion R1 =
                    util::rotateDegreeZYX(postures[low].bone_rotations[j].z(), postures[low].bone_rotations[j].y(),
                                          postures[low].bone_rotations[j].x());

                Eigen::Quaternion R2 =
                    util::rotateDegreeZYX(postures[high].bone_rotations[j].z(), postures[high].bone_rotations[j].y(),
                                          postures[high].bone_rotations[j].x());

                R1.slerp(r, R2);
                R1.normalize();
                Eigen::Vector3d temp = R1.toRotationMatrix().eulerAngles(2, 1, 0) * ((double)180 / pi);

                new_poseture.bone_rotations[j][0] = temp[0];
                new_poseture.bone_rotations[j][1] = temp[1];
                new_poseture.bone_rotations[j][2] = temp[2];
            }
        }
        new_postures.push_back(new_poseture);
    }
    return new_postures;
}
}  // namespace kinematics
