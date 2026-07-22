// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        env->robot->update();

        // Seed last_action and the first physical command from the measured
        // pose.  Without this, processed_actions is all zeros until the first
        // inference and all 29 joints receive q=0 at policy entry.
        std::vector<float> initial_action(action_scale_.size(), 0.0f);
        for (std::size_t i = 0; i < initial_action.size(); ++i)
        {
            const float scale = action_scale_[i];
            initial_action[i] = std::abs(scale) > 1.0e-6f
                ? (env->robot->data.joint_pos[i] - action_offset_[i]) / scale
                : 0.0f;
        }
        env->action_manager->process_action(initial_action);
        initial_action_ = initial_action;
        {
            std::lock_guard<std::mutex> lock(command_mutex_);
            commanded_q_ = env->action_manager->processed_actions();
        }

        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            env->reset();
            env->action_manager->process_action(initial_action_);

            std::uint32_t last_policy_tick = 0;
            {
                std::lock_guard<std::mutex> lock(lowstate->mutex_);
                last_policy_tick = lowstate->msg_.tick();
            }
            const std::uint32_t policy_period_ms = std::max<std::uint32_t>(
                1u, static_cast<std::uint32_t>(std::lround(env->step_dt * 1000.0f))
            );

            while (policy_thread_running)
            {
                std::uint32_t current_tick = 0;
                {
                    std::lock_guard<std::mutex> lock(lowstate->mutex_);
                    current_tick = lowstate->msg_.tick();
                }

                // MuJoCo keeps publishing LowState while paused, but its tick
                // does not advance.  Do not evolve last_action without a
                // corresponding physics step.
                if (current_tick < last_policy_tick &&
                    (last_policy_tick - current_tick) < 0x80000000u)
                {
                    last_policy_tick = current_tick;
                }
                if (static_cast<std::uint32_t>(current_tick - last_policy_tick)
                    < policy_period_ms)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }

                env->step();
                const auto next_q = env->action_manager->processed_actions();
                {
                    std::lock_guard<std::mutex> lock(command_mutex_);
                    commanded_q_ = next_q;
                }
                last_policy_tick = current_tick;
            }
        });
    }

    void run();
    
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    bool policy_thread_running = false;
    std::vector<float> action_scale_;
    std::vector<float> action_offset_;
    std::vector<float> initial_action_;
    std::mutex command_mutex_;
    std::vector<float> commanded_q_;
};

REGISTER_FSM(State_RLBase)
