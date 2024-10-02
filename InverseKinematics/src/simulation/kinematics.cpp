#include "simulation/kinematics.h"

#include "Eigen/Dense"
#include <iostream>
#include "acclaim/bone.h"
#include "util/helper.h"

#include <algorithm>
#include <stack>
#include <map>

namespace kinematics {

void forwardSolver(const acclaim::Posture& posture, acclaim::Bone* bone) {
    // TODO (FK)
    // Same as HW2
    // Hint:
    //   1. If you don't use `axis` in this function, you can copy-paste your code

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
                util::rotateDegreeZYX(posture.bone_rotations[temp->idx].x(), posture.bone_rotations[temp->idx].y(),
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

Eigen::VectorXd pseudoInverseLinearSolver(const Eigen::Matrix4Xd& Jacobian, const Eigen::Vector4d& target) {
    // TODO (find x which min(| jacobian * x - target |))
    // Hint:
    //   1. Linear algebra - least squares solution
    //   2. https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Construction
    // Note:
    //   1. SVD or other pseudo-inverse method is useful
    //   2. Some of them have some limitation, if you use that method you should check it.
    Eigen::VectorXd deltatheta(Jacobian.cols());
    deltatheta.setZero();

    // Compute the pseudo-inverse of Jacobian using SVD
    Eigen::MatrixXd pinvJacobian = Jacobian.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
                                       .solve(Eigen::MatrixXd::Identity(Jacobian.rows(), Jacobian.rows()));

    // Compute deltatheta that minimizes |jacobian * deltatheta - target|
    deltatheta = pinvJacobian * target;

    return deltatheta;
}

/**
 * @brief Perform inverse kinematics (IK)
 *
 * @param target_pos The position where `end_bone` will move to.
 * @param start_bone This bone is the last bone you can move while doing IK
 * @param end_bone This bone will try to reach `target_pos`
 * @param posture The original AMC motion's reference, you need to modify this
 *
 * @return True if IK is stable (HW3 bonus)
 */
bool inverseJacobianIKSolver(const Eigen::Vector4d& target_pos, acclaim::Bone* start_bone, acclaim::Bone* end_bone,
                             acclaim::Posture& posture) {
    constexpr int max_iteration = 1000;
    constexpr double epsilon = 1E-3;
    constexpr double step = 0.1;
    // Since bone stores in bones[i] that i == bone->idx, we can use bone - bone->idx to find bones[0] which is the root.
    acclaim::Bone* root_bone = start_bone - start_bone->idx;
    // TODO
    // Perform inverse kinematics (IK)
    // HINTs will tell you what should do in that area.
    // Of course you can ignore it (Any code below this line) and write your own code.
    acclaim::Posture original_posture(posture);

    size_t bone_num = 0;
    std::vector<acclaim::Bone*> boneList;
    // TODO
    // Calculate number of bones need to move to perform IK, store in `bone_num` 
    // a.k.a. how may bones from end_bone to its parent then to start_bone (include both start_bone and end_bone)
    // Store the bones need to move to perform IK into boneList
    // Hint:
    //   1. Traverse from end_bone to start_bone is easier than start to end (since there is only 1 parent)
    //   2. If start bone is not reachable from end. Go to root first.
    // Note:
    //   1. Both start_bone and end_bone should be in the list
    acclaim::Bone* current = end_bone;
   
    bool flag = false;
    for (acclaim::Bone* itr = current; itr != nullptr; itr = itr->parent) {
        if (itr->idx == start_bone->idx) flag = true;
    }

    if (flag) {
        for (acclaim::Bone* itr = current; itr->idx != start_bone->parent->idx; itr = itr->parent) {
            boneList.push_back(itr);
        }
    } else {
        for (acclaim::Bone* itr = current; itr != nullptr; itr = itr->parent) {
            boneList.push_back(itr);
        }
        for (acclaim::Bone* itr = start_bone; itr->idx != root_bone->idx; itr = itr->parent) {
            boneList.push_back(itr);
        }
    }

    bone_num = boneList.size();
    Eigen::Matrix4Xd Jacobian(4, 3 * bone_num);
    Jacobian.setZero();
    bool stable = true;
    for (int iter = 0; iter < max_iteration; ++iter) {
        forwardSolver(posture, root_bone);
        Eigen::Vector4d desiredVector = target_pos - end_bone->end_position;
        if (desiredVector.norm() < epsilon) {
            break;
        }
        // TODO (compute jacobian)
        //   1. Compute arm vectors
        //   2. Compute jacobian columns, store in `Jacobian`
        // Hint:
        //   1. You should not put rotation in jacobian if it doesn't have that DoF.
        //   2. jacobian.col(/* some column index */) = /* jacobian column */
        int column;
        for (long long i = 0; i < bone_num; i++) {
            Eigen::Vector4d dis = end_bone->end_position - boneList[i]->start_position;
            Eigen::Vector3d distance = Eigen::Vector3d(dis[0], dis[1], dis[2]);
            Eigen::Affine3d rota = boneList[i]->rotation;
            column = i * 3;

            if (boneList[i]->dofrx) {
                Eigen::Vector4d unit = rota.matrix().col(0);
                Eigen::Vector3d unit_rota = Eigen::Vector3d(unit[0], unit[1], unit[2]);
                Jacobian.col(column) =
                    Eigen::Vector4d(unit_rota.cross(distance)[0], unit_rota.cross(distance)[1], unit_rota.cross(distance)[2], 0.0);
            }

            if (boneList[i]->dofry) {
                Eigen::Vector4d unit = rota.matrix().col(1);
                Eigen::Vector3d unit_rota = Eigen::Vector3d(unit[0], unit[1], unit[2]);
                Jacobian.col(column + 1) =
                    Eigen::Vector4d(unit_rota.cross(distance)[0], unit_rota.cross(distance)[1], unit_rota.cross(distance)[2], 0.0);
            }
            
            if (boneList[i]->dofrz) {
                Eigen::Vector4d unit = rota.matrix().col(2);
                Eigen::Vector3d unit_rota = Eigen::Vector3d(unit[0], unit[1], unit[2]);
                Jacobian.col(column + 2) =
                    Eigen::Vector4d(unit_rota.cross(distance)[0], unit_rota.cross(distance)[1], unit_rota.cross(distance)[2], 0.0);
            }

        }

        Eigen::VectorXd deltatheta = step * pseudoInverseLinearSolver(Jacobian, desiredVector);

        // TODO (update rotation)
        //   Update `posture.bone_rotation` (in euler angle / degrees) using deltaTheta
        // Hint:
        //   1. You can ignore rotation limit of the bone.
        // Bonus:
        //   1. You cannot ignore rotation limit of the bone.
        

        for (long long i = 0; i < bone_num; i++) {
            acclaim::Bone bone = *boneList[i];
            Eigen::Vector4d deg = posture.bone_rotations[bone.idx] + util::toDegree(deltatheta.segment(i * 3, 3));
            bool exceedsLimit = false;

            if (deg[0] < bone.rxmin) {
                deg[0] = bone.rxmin;
                exceedsLimit = true;
            } else if (deg[0] > bone.rxmax) {
                deg[0] = bone.rxmax;
                exceedsLimit = true;
            }

            if (deg[1] < bone.rymin) {
                deg[1] = bone.rymin;
                exceedsLimit = true;
            } else if (deg[1] > bone.rymax) {
                deg[1] = bone.rymax;
                exceedsLimit = true;
            }

            if (deg[2] < bone.rzmin) {
                deg[2] = bone.rymin;
                exceedsLimit = true;
            } else if (deg[2] > bone.rzmax) {
                deg[2] = bone.rymax;
                exceedsLimit = true;
            }

            if (exceedsLimit) {
                stable = false;
            }

            posture.bone_rotations[bone.idx] = deg;

        }

    }
    // TODO (Bonus)
    // Return whether IK is stable (i.e. whether the ball is reachable) and let the skeleton not swing its hand in the air
    return stable;
}
}  // namespace kinematics
