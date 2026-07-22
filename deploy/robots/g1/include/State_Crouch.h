// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "FSM/FSMState.h"
#include "isaaclab/envs/manager_based_rl_env.h"
#include <unitree/dds_wrapper/robots/go2/go2_sub.h>

// Command-conditioned crouch controller for the physical 29-DoF G1.
//
// The state evaluates both the zero-velocity walking actor and the crouch
// actor.  It first blends between those policies at zero crouch depth, then
// increases the kinematic crouch reference over a configured duration.  This
// reproduces the hand-over used by scripts/play.py instead of abruptly
// replacing all 29 joint commands at the state boundary.
class State_Crouch : public FSMState
{
public:
    State_Crouch(int state_mode, std::string state_string);
    ~State_Crouch();

    void enter() override;
    void run() override;
    void exit() override;

private:
    void policy_loop();
    void set_policy_gains();

    std::unique_ptr<isaaclab::ManagerBasedRLEnv> walking_env_;
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> crouch_env_;
    std::shared_ptr<unitree::robot::go2::subscription::SportModeState>
        sport_state_;

    std::thread policy_thread_;
    std::atomic<bool> policy_thread_running_{false};

    std::mutex command_mutex_;
    std::vector<float> commanded_q_;

    std::vector<float> crouch_target_q_;
    std::vector<int> locked_residual_joint_ids_;
    std::vector<int> always_locked_residual_joint_ids_;
    std::vector<float> walking_action_scale_;
    std::vector<float> walking_action_offset_;

    float crouch_wait_time_s_ = 0.3f;
    float policy_blend_time_s_ = 1.0f;
    float crouch_duration_s_ = 4.0f;
    float target_depth_ = 0.90f;
    float max_joint_target_rate_rad_s_ = 3.0f;
    float lock_residual_at_depth_ = 0.75f;
    float locked_residual_scale_ = 0.20f;
    float crouch_reference_x_ = 0.0f;
    bool has_crouch_reference_x_ = false;
};

REGISTER_FSM(State_Crouch)
