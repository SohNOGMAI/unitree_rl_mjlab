// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#include "State_Crouch.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <stdexcept>

#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "unitree_articulation.h"

namespace
{
constexpr std::size_t kG1Dof = 29;
constexpr float kCrouchObservationScale = 0.1f;
constexpr float kPi = 3.14159265358979323846f;

std::atomic<float> g_crouch_depth{0.0f};
std::atomic<float> g_crouch_reference_yaw{0.0f};
std::atomic<float> g_crouch_sagittal_error{0.0f};
std::atomic<float> g_crouch_velocity_x{0.0f};
std::atomic<float> g_crouch_velocity_y{0.0f};
std::mutex g_crouch_history_mutex;
std::vector<float> g_previous_crouch_residual(kG1Dof, 0.0f);

float yaw_from_quaternion(const Eigen::Quaternionf & q)
{
    return std::atan2(
        2.0f * (q.w() * q.z() + q.x() * q.y()),
        1.0f - 2.0f * (q.y() * q.y() + q.z() * q.z())
    );
}

float wrap_angle(float angle)
{
    return std::atan2(std::sin(angle), std::cos(angle));
}

float smoothstep(float alpha)
{
    alpha = std::clamp(alpha, 0.0f, 1.0f);
    return alpha * alpha * (3.0f - 2.0f * alpha);
}
} // namespace

namespace isaaclab
{
namespace mdp
{

// The walking actor used during hand-over must receive an exact zero velocity
// command, even if the operator accidentally nudges a joystick.
REGISTER_OBSERVATION(stand_velocity_commands)
{
    return {0.0f, 0.0f, 0.0f};
}

REGISTER_OBSERVATION(stand_gait_phase)
{
    return {0.0f, 0.0f};
}

// The crouch actor keeps the original 3-D command slots:
//   [scaled crouch depth, sagittal displacement error, yaw error].
// G1's LowState supplies orientation but no drift-free world x position after
// the built-in motion service is disabled.  Therefore x error is explicitly
// zero in the sensor-free deployment, while yaw error remains observable from
// the built-in IMU.
REGISTER_OBSERVATION(crouch_command)
{
    const float yaw = yaw_from_quaternion(env->robot->data.root_quat_w);
    const float yaw_error = wrap_angle(
        yaw - g_crouch_reference_yaw.load(std::memory_order_relaxed)
    );
    return {
        g_crouch_depth.load(std::memory_order_relaxed) *
            kCrouchObservationScale,
        g_crouch_sagittal_error.load(std::memory_order_relaxed),
        yaw_error,
    };
}

// Training reused the two gait-phase slots for planar base velocity.  A
// drift-free planar velocity is not available in LowState without an external
// estimator, so both values are zero for this first proprioceptive deployment.
REGISTER_OBSERVATION(crouch_balance_state)
{
    return {
        g_crouch_velocity_x.load(std::memory_order_relaxed),
        g_crouch_velocity_y.load(std::memory_order_relaxed),
    };
}

// The trained recurrence expects the previous raw crouch residual, not the
// blended physical joint target and not the residual after joint locking.
REGISTER_OBSERVATION(crouch_last_action)
{
    std::lock_guard<std::mutex> lock(g_crouch_history_mutex);
    return g_previous_crouch_residual;
}

} // namespace mdp
} // namespace isaaclab

State_Crouch::State_Crouch(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    const auto cfg = param::config["FSM"][state_string];
    const auto walking_policy_dir = param::parser_policy_dir(
        cfg["walking_policy_dir"].as<std::string>()
    );
    const auto crouch_policy_dir = param::parser_policy_dir(
        cfg["crouch_policy_dir"].as<std::string>()
    );

    const auto walking_deploy_cfg = YAML::LoadFile(
        walking_policy_dir / "params" / "stand_deploy.yaml"
    );
    walking_env_ = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        walking_deploy_cfg,
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(
            FSMState::lowstate
        )
    );
    walking_env_->alg = std::make_unique<isaaclab::OrtRunner>(
        walking_policy_dir / "exported" / "policy.onnx"
    );
    const auto walking_action_cfg =
        walking_deploy_cfg["actions"]["JointPositionAction"];
    walking_action_scale_ =
        walking_action_cfg["scale"].as<std::vector<float>>();
    walking_action_offset_ =
        walking_action_cfg["offset"].as<std::vector<float>>();

    crouch_env_ = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(crouch_policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(
            FSMState::lowstate
        )
    );
    crouch_env_->alg = std::make_unique<isaaclab::OrtRunner>(
        crouch_policy_dir / "exported" / "policy.onnx"
    );

    // unitree_mujoco publishes its frame position/velocity on this standard
    // topic.  It reproduces play.py's crouch observations in sim-to-sim.  On
    // hardware the controller falls back to proprioceptive zeros if the
    // high-level state estimator is unavailable in low-level mode.
    sport_state_ = std::make_shared<
        unitree::robot::go2::subscription::SportModeState
    >();

    crouch_wait_time_s_ = std::max(
        cfg["crouch_wait_time_s"].as<float>(), 0.0f
    );
    policy_blend_time_s_ = std::max(
        cfg["policy_blend_time_s"].as<float>(), 0.1f
    );
    crouch_duration_s_ = std::max(
        cfg["crouch_duration_s"].as<float>(), 0.1f
    );
    target_depth_ = std::clamp(
        cfg["target_depth"].as<float>(), 0.0f, 1.0f
    );
    max_joint_target_rate_rad_s_ = std::max(
        cfg["max_joint_target_rate_rad_s"].as<float>(), 0.1f
    );
    lock_residual_at_depth_ = std::max(
        cfg["lock_residual_at_depth"].as<float>(), 1.0e-3f
    );
    locked_residual_scale_ = std::clamp(
        cfg["locked_residual_scale"].as<float>(), 0.0f, 1.0f
    );
    crouch_target_q_ = cfg["crouch_target_q"].as<std::vector<float>>();
    locked_residual_joint_ids_ =
        cfg["locked_residual_joint_ids"].as<std::vector<int>>();
    always_locked_residual_joint_ids_ =
        cfg["always_locked_residual_joint_ids"].as<std::vector<int>>();

    if (crouch_target_q_.size() != kG1Dof)
    {
        throw std::runtime_error(
            "Crouch.crouch_target_q must contain exactly 29 values."
        );
    }
    if (walking_action_scale_.size() != kG1Dof ||
        walking_action_offset_.size() != kG1Dof)
    {
        throw std::runtime_error(
            "Crouch walking action scale/offset must contain 29 values."
        );
    }
    commanded_q_.resize(kG1Dof, 0.0f);
}

State_Crouch::~State_Crouch()
{
    exit();
}

void State_Crouch::set_policy_gains()
{
    const auto & kp = walking_env_->robot->data.joint_stiffness;
    const auto & kd = walking_env_->robot->data.joint_damping;
    if (kp.size() != kG1Dof || kd.size() != kG1Dof)
    {
        throw std::runtime_error("Crouch policy gains are not 29-DoF.");
    }

    for (std::size_t i = 0; i < kG1Dof; ++i)
    {
        auto & motor = lowcmd->msg_.motor_cmd()[i];
        motor.kp() = kp[i];
        motor.kd() = kd[i];
        motor.dq() = 0.0f;
        motor.tau() = 0.0f;
    }
}

void State_Crouch::enter()
{
    set_policy_gains();
    walking_env_->reset();
    crouch_env_->reset();

    crouch_env_->robot->update();
    g_crouch_reference_yaw.store(
        yaw_from_quaternion(crouch_env_->robot->data.root_quat_w),
        std::memory_order_relaxed
    );
    g_crouch_depth.store(0.0f, std::memory_order_relaxed);
    g_crouch_sagittal_error.store(0.0f, std::memory_order_relaxed);
    g_crouch_velocity_x.store(0.0f, std::memory_order_relaxed);
    g_crouch_velocity_y.store(0.0f, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(g_crouch_history_mutex);
        std::fill(
            g_previous_crouch_residual.begin(),
            g_previous_crouch_residual.end(),
            0.0f
        );
    }

    // Preserve the final walking-policy target.  Replacing it by the measured
    // angle would zero the supporting PD error/torque at the state boundary.
    std::vector<float> initial_walking_action(kG1Dof, 0.0f);
    {
        std::lock_guard<std::mutex> command_lock(command_mutex_);
        for (std::size_t i = 0; i < kG1Dof; ++i)
        {
            commanded_q_[i] = lowcmd->msg_.motor_cmd()[i].q();
            if (!std::isfinite(commanded_q_[i]))
            {
                std::lock_guard<std::mutex> state_lock(lowstate->mutex_);
                commanded_q_[i] = lowstate->msg_.motor_state()[i].q();
            }
            lowcmd->msg_.motor_cmd()[i].q() = commanded_q_[i];
            initial_walking_action[i] =
                (commanded_q_[i] - walking_action_offset_[i]) /
                walking_action_scale_[i];
        }
    }
    walking_env_->action_manager->process_action(initial_walking_action);

    has_crouch_reference_x_ = false;
    if (sport_state_ && !sport_state_->isTimeout())
    {
        std::lock_guard<std::mutex> lock(sport_state_->mutex_);
        crouch_reference_x_ = sport_state_->msg_.position()[0];
        has_crouch_reference_x_ = true;
    }

    policy_thread_running_.store(true);
    policy_thread_ = std::thread(&State_Crouch::policy_loop, this);
    spdlog::info(
        "Crouch started: {:.1f}s wait, {:.1f}s policy blend, {:.1f}s depth ramp to {:.2f}.",
        crouch_wait_time_s_,
        policy_blend_time_s_,
        crouch_duration_s_,
        target_depth_
    );
}

void State_Crouch::run()
{
    std::lock_guard<std::mutex> lock(command_mutex_);
    for (std::size_t i = 0; i < commanded_q_.size(); ++i)
    {
        lowcmd->msg_.motor_cmd()[i].q() = commanded_q_[i];
    }
}

void State_Crouch::exit()
{
    policy_thread_running_.store(false);
    if (policy_thread_.joinable())
    {
        policy_thread_.join();
    }
}

void State_Crouch::policy_loop()
{
    const float step_dt = crouch_env_->step_dt;
    const std::uint32_t policy_period_ms = std::max<std::uint32_t>(
        1u, static_cast<std::uint32_t>(std::lround(step_dt * 1000.0f))
    );
    std::uint32_t start_tick = 0;
    {
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        start_tick = lowstate->msg_.tick();
    }
    std::uint32_t last_policy_tick = start_tick;
    std::vector<float> previous_q;
    {
        std::lock_guard<std::mutex> lock(command_mutex_);
        previous_q = commanded_q_;
    }
    std::size_t log_counter = 0;

    while (policy_thread_running_.load())
    {
        std::uint32_t current_tick = 0;
        {
            std::lock_guard<std::mutex> lock(lowstate->mutex_);
            current_tick = lowstate->msg_.tick();
        }
        if (current_tick < last_policy_tick &&
            (last_policy_tick - current_tick) < 0x80000000u)
        {
            start_tick = current_tick;
            last_policy_tick = current_tick;
        }
        if (static_cast<std::uint32_t>(current_tick - last_policy_tick)
            < policy_period_ms)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        last_policy_tick = current_tick;

        const float elapsed_s =
            static_cast<std::uint32_t>(current_tick - start_tick) * 1.0e-3f;
        const float handover_elapsed = std::max(
            elapsed_s - crouch_wait_time_s_, 0.0f
        );
        const float blend = smoothstep(
            handover_elapsed / policy_blend_time_s_
        );
        const float depth_alpha = std::clamp(
            (handover_elapsed - policy_blend_time_s_) / crouch_duration_s_,
            0.0f,
            1.0f
        );
        const float depth = target_depth_ * depth_alpha;
        // The fine-tuned crouch actor is a residual controller around a
        // nonzero crouch command; at depth zero its raw output is not the
        // standing actor.  Warm its recurrent action history during handover,
        // but give it physical authority only in proportion to crouch depth.
        // This makes the commanded joint target exactly continuous at depth 0.
        const float effective_blend = blend * (
            target_depth_ > 1.0e-6f ? depth / target_depth_ : 0.0f
        );
        g_crouch_depth.store(depth, std::memory_order_relaxed);

        // Zero-command walking actor.  This provides the same stable standing
        // controller used immediately before entering Crouch.
        walking_env_->robot->update();
        const auto walking_obs = walking_env_->observation_manager->compute();
        const auto walking_action = walking_env_->alg->act(walking_obs);
        walking_env_->action_manager->process_action(walking_action);
        const auto walking_q =
            walking_env_->action_manager->processed_actions();

        // Command-conditioned crouch actor and its independent action history.
        crouch_env_->robot->update();
        if (sport_state_ && !sport_state_->isTimeout())
        {
            Eigen::Vector3f position_w;
            Eigen::Vector3f velocity_w;
            {
                std::lock_guard<std::mutex> lock(sport_state_->mutex_);
                position_w = Eigen::Map<const Eigen::Vector3f>(
                    sport_state_->msg_.position().data()
                );
                velocity_w = Eigen::Map<const Eigen::Vector3f>(
                    sport_state_->msg_.velocity().data()
                );
            }
            if (!has_crouch_reference_x_)
            {
                crouch_reference_x_ = position_w.x();
                has_crouch_reference_x_ = true;
            }
            const Eigen::Vector3f velocity_b =
                crouch_env_->robot->data.root_quat_w.conjugate() * velocity_w;
            g_crouch_sagittal_error.store(
                2.0f * (position_w.x() - crouch_reference_x_),
                std::memory_order_relaxed
            );
            g_crouch_velocity_x.store(
                0.2f * velocity_b.x(), std::memory_order_relaxed
            );
            g_crouch_velocity_y.store(
                0.2f * velocity_b.y(), std::memory_order_relaxed
            );
        }
        const auto crouch_obs = crouch_env_->observation_manager->compute();
        const auto raw_crouch_action = crouch_env_->alg->act(crouch_obs);
        {
            std::lock_guard<std::mutex> lock(g_crouch_history_mutex);
            g_previous_crouch_residual = raw_crouch_action;
        }

        auto adjusted_crouch_action = raw_crouch_action;
        const float lock_blend = std::clamp(
            depth / lock_residual_at_depth_, 0.0f, 1.0f
        );
        const float residual_scale =
            1.0f - lock_blend * (1.0f - locked_residual_scale_);
        for (const int id : locked_residual_joint_ids_)
        {
            if (id >= 0 && static_cast<std::size_t>(id) < kG1Dof)
            {
                adjusted_crouch_action[id] *= residual_scale;
            }
        }
        for (const int id : always_locked_residual_joint_ids_)
        {
            if (id >= 0 && static_cast<std::size_t>(id) < kG1Dof)
            {
                adjusted_crouch_action[id] = 0.0f;
            }
        }
        crouch_env_->action_manager->process_action(adjusted_crouch_action);
        auto crouch_q = crouch_env_->action_manager->processed_actions();

        if (walking_q.size() != kG1Dof || crouch_q.size() != kG1Dof)
        {
            spdlog::critical("Walking/crouch actor returned a non-29D action.");
            policy_thread_running_.store(false);
            break;
        }

        const auto & default_q = crouch_env_->robot->data.default_joint_pos;
        const float max_delta = max_joint_target_rate_rad_s_ * step_dt;
        std::vector<float> next_q(kG1Dof, 0.0f);
        bool valid = true;
        for (std::size_t i = 0; i < kG1Dof; ++i)
        {
            // Add the same command-interpolated physical reference used by
            // CrouchResidualJointPositionAction during training.
            crouch_q[i] += depth * (crouch_target_q_[i] - default_q[i]);
            const float requested =
                (1.0f - effective_blend) * walking_q[i] +
                effective_blend * crouch_q[i];
            if (!std::isfinite(requested))
            {
                valid = false;
                break;
            }
            next_q[i] = previous_q[i] + std::clamp(
                requested - previous_q[i], -max_delta, max_delta
            );
        }

        if (valid)
        {
            {
                std::lock_guard<std::mutex> lock(command_mutex_);
                commanded_q_ = next_q;
            }
            previous_q = std::move(next_q);
        }
        else
        {
            spdlog::error(
                "Crouch actor produced NaN/Inf; holding the previous command."
            );
        }

        if ((log_counter++ % 50) == 0)
        {
            const float current_yaw = yaw_from_quaternion(
                crouch_env_->robot->data.root_quat_w
            );
            const float yaw_error_deg = wrap_angle(
                current_yaw -
                g_crouch_reference_yaw.load(std::memory_order_relaxed)
            ) * 180.0f / kPi;
            const char * phase = elapsed_s < policy_blend_time_s_
                ? "POLICY_BLEND"
                : (depth_alpha < 1.0f ? "CROUCHING" : "HOLD");
            spdlog::info(
                "Crouch phase={} warmup={:.2f} applied={:.2f} depth={:.2f} yaw_error={:+.1f}deg",
                phase,
                blend,
                effective_blend,
                depth,
                yaw_error_deg
            );
        }

    }
}
