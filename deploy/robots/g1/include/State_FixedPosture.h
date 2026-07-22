// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "FSM/FSMState.h"

// Holds the experimentally selected ascend suspension pose.  This state does
// not balance the robot: it is intended for a robot already supported by the
// safety harness/wire.  Entry is rate-smoothed from measured joint positions.
class State_FixedPosture : public FSMState
{
public:
    State_FixedPosture(int state_mode, std::string state_string)
    : FSMState(state_mode, state_string)
    {
        const auto cfg = param::config["FSM"][state_string];
        duration_s_ = std::max(cfg["duration_s"].as<float>(), 0.1f);
        kp_ = cfg["kp"].as<std::vector<float>>();
        kd_ = cfg["kd"].as<std::vector<float>>();
        target_q_ = cfg["target_q"].as<std::vector<float>>();

        constexpr std::size_t kG1Dof = 29;
        if (kp_.size() != kG1Dof || kd_.size() != kG1Dof ||
            target_q_.size() != kG1Dof)
        {
            throw std::runtime_error(
                "FixedPosture expects exactly 29 kp, kd, and target_q values."
            );
        }
        start_q_.resize(kG1Dof, 0.0f);
    }

    void enter() override
    {
        // Start from measured joint positions.  Starting from a stale command
        // can create an instantaneous jump when entering this state.
        {
            std::lock_guard<std::mutex> lock(lowstate->mutex_);
            for (std::size_t i = 0; i < start_q_.size(); ++i)
            {
                start_q_[i] = lowstate->msg_.motor_state()[i].q();
            }
        }

        for (std::size_t i = 0; i < target_q_.size(); ++i)
        {
            auto & motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp() = kp_[i];
            motor.kd() = kd_[i];
            motor.dq() = 0.0f;
            motor.tau() = 0.0f;
            motor.q() = start_q_[i];
        }

        start_time_s_ = static_cast<double>(
            unitree::common::GetCurrentTimeMillisecond()
        ) * 1.0e-3;
        spdlog::warn(
            "FixedPosture entered. Keep the robot supported by the harness."
        );
    }

    void run() override
    {
        const double now_s = static_cast<double>(
            unitree::common::GetCurrentTimeMillisecond()
        ) * 1.0e-3;
        const float alpha = std::clamp(
            static_cast<float>((now_s - start_time_s_) / duration_s_),
            0.0f,
            1.0f
        );
        // Smoothstep has zero slope at both ends, unlike a linear ramp.
        const float blend = alpha * alpha * (3.0f - 2.0f * alpha);

        for (std::size_t i = 0; i < target_q_.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].q() =
                (1.0f - blend) * start_q_[i] + blend * target_q_[i];
        }
    }

private:
    float duration_s_ = 2.0f;
    double start_time_s_ = 0.0;
    std::vector<float> kp_;
    std::vector<float> kd_;
    std::vector<float> start_q_;
    std::vector<float> target_q_;
};

REGISTER_FSM(State_FixedPosture)
