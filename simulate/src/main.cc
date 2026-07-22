// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// !!! hack code: make glfw_adapter.window_ public
#define private public
#include "glfw_adapter.h"
#undef private

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>

#include <mujoco/mujoco.h>
#include "simulate.h"
#include "array_safety.h"
#include "unitree_sdk2_bridge.h"
#include "param.h"

#define MUJOCO_PLUGIN_DIR "mujoco_plugin"
#define NUM_MOTOR_IDL_GO 20

extern "C"
{
#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#else
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <sys/errno.h>
#include <unistd.h>
#endif
}

class ElasticBand
{
public:
  ElasticBand(){};
  void Advance(std::vector<double> x, std::vector<double> dx)
  {
    std::vector<double> delta_x = {0.0, 0.0, 0.0};
    delta_x[0] = point_[0] - x[0];
    delta_x[1] = point_[1] - x[1];
    delta_x[2] = point_[2] - x[2];
    double distance = sqrt(delta_x[0] * delta_x[0] + delta_x[1] * delta_x[1] + delta_x[2] * delta_x[2]);

    std::vector<double> direction = {0.0, 0.0, 0.0};
    direction[0] = delta_x[0] / distance;
    direction[1] = delta_x[1] / distance;
    direction[2] = delta_x[2] / distance;

    double v = dx[0] * direction[0] + dx[1] * direction[1] + dx[2] * direction[2];

    f_[0] = (stiffness_ * (distance - length_) - damping_ * v) * direction[0];
    f_[1] = (stiffness_ * (distance - length_) - damping_ * v) * direction[1];
    f_[2] = (stiffness_ * (distance - length_) - damping_ * v) * direction[2];
  }


  double stiffness_ = 200;
  double damping_ = 100;
  std::vector<double> point_ = {0, 0, 3};
  double length_ = 0.0;
  bool enable_ = true;
  std::vector<double> f_ = {0, 0, 0};
};
inline ElasticBand elastic_band;


namespace
{
  namespace mj = ::mujoco;
  namespace mju = ::mujoco::sample_util;

  // constants
  const double syncMisalign = 0.1;       // maximum mis-alignment before re-sync (simulation seconds)
  const double simRefreshFraction = 0.7; // fraction of refresh available for simulation
  const int kErrorLength = 1024;         // load error string length

  // model and data
  mjModel *m = nullptr;
  mjData *d = nullptr;

  class VirtualHoist
  {
  public:
    void Apply()
    {
      if (!param::config.virtual_hoist || !m || !d)
      {
        return;
      }

      if (body_id_ < 0 || site_id_ < 0)
      {
        body_id_ = mj_name2id(m, mjOBJ_BODY, "torso_link");
        site_id_ = mj_name2id(m, mjOBJ_SITE, "wire_attachment");
        if (body_id_ < 0 || site_id_ < 0)
        {
          if (!missing_site_reported_)
          {
            std::cerr << "[HOIST] torso_link or wire_attachment was not found; "
                         "virtual hoist disabled\n";
            missing_site_reported_ = true;
          }
          return;
        }
      }

      // Do not leave a stale wrench behind after reset or before start.
      double* wrench = d->xfrc_applied + 6 * body_id_;
      for (int i = 0; i < 6; ++i) wrench[i] = 0.0;

      if (d->time < last_time_s_)
      {
        Reset();
      }
      const double dt = last_time_s_ >= 0.0
          ? std::max(d->time - last_time_s_, 0.0)
          : m->opt.timestep;
      last_time_s_ = d->time;

      const bool start_requested = param::config.virtual_hoist_gamepad_trigger
          ? virtual_hoist_requested.load(std::memory_order_acquire)
          : d->time >= param::config.virtual_hoist_start_s;
      if (!start_requested)
      {
        return;
      }

      const double* p = d->site_xpos + 3 * site_id_;
      const double anchor[3] = {
          param::config.virtual_hoist_anchor_x,
          param::config.virtual_hoist_anchor_y,
          param::config.virtual_hoist_anchor_z};
      double delta[3] = {
          anchor[0] - p[0], anchor[1] - p[1], anchor[2] - p[2]};
      const double length = mju_norm3(delta);
      if (length < 1.0e-6) return;
      double direction[3] = {
          delta[0] / length, delta[1] / length, delta[2] / length};

      if (!armed_)
      {
        // Capture the measured length first.  Consequently enabling the cable
        // produces zero extension and no impulsive preload.
        initial_length_m_ = length;
        commanded_length_m_ = length;
        const double* com = d->xpos + 3 * body_id_;
        guide_x_m_ = com[0];
        guide_y_m_ = com[1];
        const double* rotation = d->xmat + 9 * body_id_;
        guide_yaw_rad_ = std::atan2(rotation[3], rotation[0]);
        armed_ = true;
        std::cout << std::fixed << std::setprecision(3)
                  << "[HOIST] armed at t=" << d->time
                  << "s L0=" << initial_length_m_ << "m\n";
      }

      const double target_length = std::max(
          initial_length_m_ - param::config.virtual_hoist_reel_distance_m, 0.05);
      commanded_length_m_ = std::max(
          target_length,
          commanded_length_m_ - param::config.virtual_hoist_reel_speed_m_s * dt);

      mjtNum velocity6[6] = {0, 0, 0, 0, 0, 0};
      mj_objectVelocity(m, d, mjOBJ_SITE, site_id_, velocity6, 0);
      // MuJoCo returns angular velocity first, then linear velocity.
      const double toward_anchor_velocity =
          velocity6[3] * direction[0] +
          velocity6[4] * direction[1] +
          velocity6[5] * direction[2];
      const double length_rate = -toward_anchor_velocity;
      const double raw_tension = std::clamp(
          param::config.virtual_hoist_stiffness_n_m *
              (length - commanded_length_m_) +
          param::config.virtual_hoist_damping_n_s_m * length_rate,
          0.0, param::config.virtual_hoist_max_tension_n);
      const double max_delta_tension =
          param::config.virtual_hoist_tension_rate_n_s * dt;
      tension_n_ += std::clamp(
          raw_tension - tension_n_, -max_delta_tension, max_delta_tension);

      const double force[3] = {
          tension_n_ * direction[0],
          tension_n_ * direction[1],
          tension_n_ * direction[2]};
      const double* com = d->xpos + 3 * body_id_;
      const double r[3] = {p[0] - com[0], p[1] - com[1], p[2] - com[2]};
      const double torque[3] = {
          r[1] * force[2] - r[2] * force[1],
          r[2] * force[0] - r[0] * force[2],
          r[0] * force[1] - r[1] * force[0]};
      for (int i = 0; i < 3; ++i)
      {
        wrench[i] = force[i];
        wrench[3 + i] = torque[i];
      }

      if (param::config.virtual_hoist_pose_guide)
      {
        // Test fixture only: it removes horizontal pendulum motion and cable-
        // axis spin while leaving vertical translation completely free.  This
        // isolates the joint-posture transition from single-cable yaw physics.
        mjtNum body_velocity[6] = {0, 0, 0, 0, 0, 0};
        mj_objectVelocity(m, d, mjOBJ_BODY, body_id_, body_velocity, 0);
        constexpr double kPosition = 400.0;
        constexpr double dPosition = 120.0;
        constexpr double maxHorizontalForce = 150.0;
        double guide_fx = kPosition * (guide_x_m_ - com[0])
                        - dPosition * body_velocity[3];
        double guide_fy = kPosition * (guide_y_m_ - com[1])
                        - dPosition * body_velocity[4];
        const double guide_force_norm = std::hypot(guide_fx, guide_fy);
        if (guide_force_norm > maxHorizontalForce)
        {
          const double scale = maxHorizontalForce / guide_force_norm;
          guide_fx *= scale;
          guide_fy *= scale;
        }
        wrench[0] += guide_fx;
        wrench[1] += guide_fy;

        const double* rotation = d->xmat + 9 * body_id_;
        const double yaw = std::atan2(rotation[3], rotation[0]);
        const double yaw_error = std::atan2(
            std::sin(guide_yaw_rad_ - yaw),
            std::cos(guide_yaw_rad_ - yaw));
        constexpr double kYaw = 30.0;
        constexpr double dYaw = 10.0;
        constexpr double maxYawTorque = 30.0;
        wrench[5] += std::clamp(
            kYaw * yaw_error - dYaw * body_velocity[2],
            -maxYawTorque, maxYawTorque);
      }

      if (d->time >= next_log_time_s_)
      {
        std::cout << std::fixed << std::setprecision(3)
                  << "[HOIST] t=" << d->time << "s L=" << length
                  << "m Lcmd=" << commanded_length_m_
                  << "m T=" << std::setprecision(1) << tension_n_
                  << "N z_attach=" << std::setprecision(3) << p[2] << "m\n";
        next_log_time_s_ = d->time + 0.25;
      }
    }

  private:
    void Reset()
    {
      armed_ = false;
      initial_length_m_ = 0.0;
      commanded_length_m_ = 0.0;
      tension_n_ = 0.0;
      last_time_s_ = -1.0;
      next_log_time_s_ = 0.0;
      guide_x_m_ = 0.0;
      guide_y_m_ = 0.0;
      guide_yaw_rad_ = 0.0;
    }

    int body_id_ = -1;
    int site_id_ = -1;
    bool missing_site_reported_ = false;
    bool armed_ = false;
    double initial_length_m_ = 0.0;
    double commanded_length_m_ = 0.0;
    double tension_n_ = 0.0;
    double last_time_s_ = -1.0;
    double next_log_time_s_ = 0.0;
    double guide_x_m_ = 0.0;
    double guide_y_m_ = 0.0;
    double guide_yaw_rad_ = 0.0;
  };

  VirtualHoist virtual_hoist;

  // control noise variables
  mjtNum *ctrlnoise = nullptr;

  using Seconds = std::chrono::duration<double>;

  //---------------------------------------- plugin handling -----------------------------------------

  // return the path to the directory containing the current executable
  // used to determine the location of auto-loaded plugin libraries
  std::string getExecutableDir()
  {
#if defined(_WIN32) || defined(__CYGWIN__)
    constexpr char kPathSep = '\\';
    std::string realpath = [&]() -> std::string
    {
      std::unique_ptr<char[]> realpath(nullptr);
      DWORD buf_size = 128;
      bool success = false;
      while (!success)
      {
        realpath.reset(new (std::nothrow) char[buf_size]);
        if (!realpath)
        {
          std::cerr << "cannot allocate memory to store executable path\n";
          return "";
        }

        DWORD written = GetModuleFileNameA(nullptr, realpath.get(), buf_size);
        if (written < buf_size)
        {
          success = true;
        }
        else if (written == buf_size)
        {
          // realpath is too small, grow and retry
          buf_size *= 2;
        }
        else
        {
          std::cerr << "failed to retrieve executable path: " << GetLastError() << "\n";
          return "";
        }
      }
      return realpath.get();
    }();
#else
    constexpr char kPathSep = '/';
#if defined(__APPLE__)
    std::unique_ptr<char[]> buf(nullptr);
    {
      std::uint32_t buf_size = 0;
      _NSGetExecutablePath(nullptr, &buf_size);
      buf.reset(new char[buf_size]);
      if (!buf)
      {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }
      if (_NSGetExecutablePath(buf.get(), &buf_size))
      {
        std::cerr << "unexpected error from _NSGetExecutablePath\n";
      }
    }
    const char *path = buf.get();
#else
    const char *path = "/proc/self/exe";
#endif
    std::string realpath = [&]() -> std::string
    {
      std::unique_ptr<char[]> realpath(nullptr);
      std::uint32_t buf_size = 128;
      bool success = false;
      while (!success)
      {
        realpath.reset(new (std::nothrow) char[buf_size]);
        if (!realpath)
        {
          std::cerr << "cannot allocate memory to store executable path\n";
          return "";
        }

        std::size_t written = readlink(path, realpath.get(), buf_size);
        if (written < buf_size)
        {
          realpath.get()[written] = '\0';
          success = true;
        }
        else if (written == -1)
        {
          if (errno == EINVAL)
          {
            // path is already not a symlink, just use it
            return path;
          }

          std::cerr << "error while resolving executable path: " << strerror(errno) << '\n';
          return "";
        }
        else
        {
          // realpath is too small, grow and retry
          buf_size *= 2;
        }
      }
      return realpath.get();
    }();
#endif

    if (realpath.empty())
    {
      return "";
    }

    for (std::size_t i = realpath.size() - 1; i > 0; --i)
    {
      if (realpath.c_str()[i] == kPathSep)
      {
        return realpath.substr(0, i);
      }
    }

    // don't scan through the entire file system's root
    return "";
  }

  // scan for libraries in the plugin directory to load additional plugins
  void scanPluginLibraries()
  {
    // check and print plugins that are linked directly into the executable
    int nplugin = mjp_pluginCount();
    if (nplugin)
    {
      std::printf("Built-in plugins:\n");
      for (int i = 0; i < nplugin; ++i)
      {
        std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
      }
    }

    // define platform-specific strings
#if defined(_WIN32) || defined(__CYGWIN__)
    const std::string sep = "\\";
#else
    const std::string sep = "/";
#endif

    // try to open the ${EXECDIR}/plugin directory
    // ${EXECDIR} is the directory containing the simulate binary itself
    const std::string executable_dir = getExecutableDir();
    if (executable_dir.empty())
    {
      return;
    }

    const std::string plugin_dir = getExecutableDir() + sep + MUJOCO_PLUGIN_DIR;
    mj_loadAllPluginLibraries(
        plugin_dir.c_str(), +[](const char *filename, int first, int count)
                            {
        std::printf("Plugins registered by library '%s':\n", filename);
        for (int i = first; i < first + count; ++i) {
          std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
        } });
  }

  //------------------------------------------- simulation -------------------------------------------

  mjModel *LoadModel(const char *file, mj::Simulate &sim)
  {
    // this copy is needed so that the mju::strlen call below compiles
    char filename[mj::Simulate::kMaxFilenameLength];
    mju::strcpy_arr(filename, file);

    // make sure filename is not empty
    if (!filename[0])
    {
      return nullptr;
    }

    // load and compile
    char loadError[kErrorLength] = "";
    mjModel *mnew = 0;
    if (mju::strlen_arr(filename) > 4 &&
        !std::strncmp(filename + mju::strlen_arr(filename) - 4, ".mjb",
                      mju::sizeof_arr(filename) - mju::strlen_arr(filename) + 4))
    {
      mnew = mj_loadModel(filename, nullptr);
      if (!mnew)
      {
        mju::strcpy_arr(loadError, "could not load binary model");
      }
    }
    else
    {
      mnew = mj_loadXML(filename, nullptr, loadError, kErrorLength);
      // remove trailing newline character from loadError
      if (loadError[0])
      {
        int error_length = mju::strlen_arr(loadError);
        if (loadError[error_length - 1] == '\n')
        {
          loadError[error_length - 1] = '\0';
        }
      }
    }

    mju::strcpy_arr(sim.load_error, loadError);

    if (!mnew)
    {
      std::printf("%s\n", loadError);
      return nullptr;
    }

    // compiler warning: print and pause
    if (loadError[0])
    {
      // mj_forward() below will print the warning message
      std::printf("Model compiled, but simulation warning (paused):\n  %s\n", loadError);
      sim.run = 0;
    }

    return mnew;
  }

  // simulate in background thread (while rendering in main thread)
  void PhysicsLoop(mj::Simulate &sim)
  {
    // cpu-sim syncronization point
    std::chrono::time_point<mj::Simulate::Clock> syncCPU;
    mjtNum syncSim = 0;
    bool auto_start_reported = false;
    bool controller_command_detected = false;
    std::chrono::steady_clock::time_point controller_command_detected_at;
    double next_diagnostic_time = 0.0;

    // ChannelFactory::Instance()->Init(0);
    // UnitreeDds ud(d);

    // run until asked to exit
    while (!sim.exitrequest.load())
    {
      if (sim.droploadrequest.load())
      {
        sim.LoadMessage(sim.dropfilename);
        mjModel *mnew = LoadModel(sim.dropfilename, sim);
        sim.droploadrequest.store(false);

        mjData *dnew = nullptr;
        if (mnew)
          dnew = mj_makeData(mnew);
        if (dnew)
        {
          sim.Load(mnew, dnew, sim.dropfilename);

          mj_deleteData(d);
          mj_deleteModel(m);

          m = mnew;
          d = dnew;
          mj_forward(m, d);

          // allocate ctrlnoise
          free(ctrlnoise);
          ctrlnoise = (mjtNum *)malloc(sizeof(mjtNum) * m->nu);
          mju_zero(ctrlnoise, m->nu);
        }
        else
        {
          sim.LoadMessageClear();
        }
      }

      if (sim.uiloadrequest.load())
      {
        sim.uiloadrequest.fetch_sub(1);
        sim.LoadMessage(sim.filename);
        mjModel *mnew = LoadModel(sim.filename, sim);
        mjData *dnew = nullptr;
        if (mnew)
          dnew = mj_makeData(mnew);
        if (dnew)
        {
          sim.Load(mnew, dnew, sim.filename);

          mj_deleteData(d);
          mj_deleteModel(m);

          m = mnew;
          d = dnew;
          mj_forward(m, d);

          // allocate ctrlnoise
          free(ctrlnoise);
          ctrlnoise = static_cast<mjtNum *>(malloc(sizeof(mjtNum) * m->nu));
          mju_zero(ctrlnoise, m->nu);
        }
        else
        {
          sim.LoadMessageClear();
        }
      }

      // sleep for 1 ms or yield, to let main thread run
      //  yield results in busy wait - which has better timing but kills battery life
      if (sim.run && sim.busywait)
      {
        std::this_thread::yield();
      }
      else
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }

      {
        // lock the sim mutex
        const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

        // run only if model is present
        if (m)
        {
          // running
          if (sim.run)
          {
            bool stepped = false;

            // record cpu time at start of iteration
            const auto startCPU = mj::Simulate::Clock::now();

            // elapsed CPU and simulation time since last sync
            const auto elapsedCPU = startCPU - syncCPU;
            double elapsedSim = d->time - syncSim;

            // inject noise
            if (sim.ctrl_noise_std)
            {
              // convert rate and scale to discrete time (Ornstein–Uhlenbeck)
              mjtNum rate = mju_exp(-m->opt.timestep / mju_max(sim.ctrl_noise_rate, mjMINVAL));
              mjtNum scale = sim.ctrl_noise_std * mju_sqrt(1 - rate * rate);

              for (int i = 0; i < m->nu; i++)
              {
                // update noise
                ctrlnoise[i] = rate * ctrlnoise[i] + scale * mju_standardNormal(nullptr);

                // apply noise
                d->ctrl[i] = ctrlnoise[i];
              }
            }

            // requested slow-down factor
            double slowdown = 100 / sim.percentRealTime[sim.real_time_index];

            // misalignment condition: distance from target sim time is bigger than syncmisalign
            bool misaligned =
                mju_abs(Seconds(elapsedCPU).count() / slowdown - elapsedSim) > syncMisalign;

            // out-of-sync (for any reason): reset sync times, step
            if (elapsedSim < 0 || elapsedCPU.count() < 0 || syncCPU.time_since_epoch().count() == 0 ||
                misaligned || sim.speed_changed)
            {
              // re-sync
              syncCPU = startCPU;
              syncSim = d->time;
              sim.speed_changed = false;

              // run single step, let next iteration deal with timing
              virtual_hoist.Apply();
              mj_step(m, d);
              stepped = true;
            }

            // in-sync: step until ahead of cpu
            else
            {
              bool measured = false;
              mjtNum prevSim = d->time;

              double refreshTime = simRefreshFraction / sim.refresh_rate;

              // step while sim lags behind cpu and within refreshTime
              while (Seconds((d->time - syncSim) * slowdown) < mj::Simulate::Clock::now() - syncCPU &&
                     mj::Simulate::Clock::now() - startCPU < Seconds(refreshTime))
              {
                // measure slowdown before first step
                if (!measured && elapsedSim)
                {
                  sim.measured_slowdown =
                      std::chrono::duration<double>(elapsedCPU).count() / elapsedSim;
                  measured = true;
                }

                // elastic band on base link
                if (param::config.enable_elastic_band == 1)
                {
                  if (elastic_band.enable_)
                  {
                    std::vector<double> x = {d->qpos[0], d->qpos[1], d->qpos[2]};
                    std::vector<double> dx = {d->qvel[0], d->qvel[1], d->qvel[2]};

                    elastic_band.Advance(x, dx);

                    d->xfrc_applied[param::config.band_attached_link] = elastic_band.f_[0];
                    d->xfrc_applied[param::config.band_attached_link + 1] = elastic_band.f_[1];
                    d->xfrc_applied[param::config.band_attached_link + 2] = elastic_band.f_[2];
                  }
                }

                // Apply the simulation-only cable immediately before every
                // integration step so its force cannot depend on viewer FPS.
                virtual_hoist.Apply();

                // call mj_step
                mj_step(m, d);
                stepped = true;

                // break if reset
                if (d->time < prevSim)
                {
                  break;
                }
              }
            }

            // save current state to history buffer
            if (stepped)
            {
              sim.AddToHistory();
            }
          }

          // paused
          else
          {
            // run mj_forward, to update rendering and joint sliders
            mj_forward(m, d);
            sim.speed_changed = true;
            if (param::config.auto_start_on_control &&
                lowlevel_controller_armed.load(std::memory_order_acquire) &&
                !controller_command_detected)
            {
              controller_command_detected = true;
              controller_command_detected_at = std::chrono::steady_clock::now();
            }
            const bool controller_settled = controller_command_detected &&
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - controller_command_detected_at
                ).count() >= param::config.auto_start_control_delay_s;
            if (param::config.auto_start_on_control && controller_settled)
            {
              sim.run = 1;
              if (!auto_start_reported)
              {
                std::cout << "[SIM] Controller armed; physics started automatically."
                          << std::endl;
                auto_start_reported = true;
              }
            }
          }

          if (param::config.print_robot_diagnostics && sim.run && d->time >= next_diagnostic_time)
          {
            // The G1 root is a free joint: xyz followed by a wxyz quaternion.
            const double w = d->qpos[3];
            const double x = d->qpos[4];
            const double y = d->qpos[5];
            const double z = d->qpos[6];
            const double roll = std::atan2(
                2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
            const double pitch = std::asin(std::clamp(
                2.0 * (w * y - z * x), -1.0, 1.0));
            const double yaw = std::atan2(
                2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
            constexpr double rad_to_deg = 57.29577951308232;
            std::cout << std::fixed << std::setprecision(3)
                      << "[SIM] t=" << d->time
                      << " pos=(" << d->qpos[0] << "," << d->qpos[1] << "," << d->qpos[2] << ")"
                      << " rpy_deg=(" << roll * rad_to_deg << ","
                      << pitch * rad_to_deg << "," << yaw * rad_to_deg << ")"
                      << std::endl;
            next_diagnostic_time = d->time + 0.25;
          }
        }
      } // release std::lock_guard<std::mutex>
    }
  }
} // namespace

//-------------------------------------- physics_thread --------------------------------------------

void PhysicsThread(mj::Simulate *sim, const char *filename)
{
  // request loadmodel if file given (otherwise drag-and-drop)
  if (filename != nullptr)
  {
    sim->LoadMessage(filename);
    m = LoadModel(filename, *sim);
    if (m)
      d = mj_makeData(m);
    if (d)
    {
      if (!param::config.initial_keyframe.empty())
      {
        int key_id = mj_name2id(m, mjOBJ_KEY, param::config.initial_keyframe.c_str());
        if (key_id < 0)
        {
          std::cerr << "Initial keyframe not found: "
                    << param::config.initial_keyframe << std::endl;
          std::exit(EXIT_FAILURE);
        }
        mj_resetDataKeyframe(m, d, key_id);
        std::cout << "Initial keyframe loaded: "
                  << param::config.initial_keyframe << std::endl;
      }
      sim->Load(m, d, filename);
      mj_forward(m, d);

      // allocate ctrlnoise
      free(ctrlnoise);
      ctrlnoise = static_cast<mjtNum *>(malloc(sizeof(mjtNum) * m->nu));
      mju_zero(ctrlnoise, m->nu);
    }
    else
    {
      sim->LoadMessageClear();
    }
  }

  PhysicsLoop(*sim);

  // delete everything we allocated
  free(ctrlnoise);
  mj_deleteData(d);
  mj_deleteModel(m);

  exit(0);
}

void *UnitreeSdk2BridgeThread(void *arg)
{
  // Wait for mujoco data
  while (true)
  {
    if (d)
    {
      std::cout << "Mujoco data is prepared" << std::endl;
      break;
    }
    usleep(500000);
  }

  unitree::robot::ChannelFactory::Instance()->Init(param::config.domain_id, param::config.interface);


  int body_id = mj_name2id(m, mjOBJ_BODY, "torso_link");
  if (body_id < 0) {
    body_id = mj_name2id(m, mjOBJ_BODY, "base_link");
  }
  param::config.band_attached_link = 6 * body_id;
  
  std::unique_ptr<UnitreeSDK2BridgeBase> interface = nullptr;
  if (m->nu > NUM_MOTOR_IDL_GO) {
    interface = std::make_unique<G1Bridge>(m, d);
  } else {
    interface = std::make_unique<Go2Bridge>(m, d);
  }
  interface->start();
  
  while (true)
  {
    sleep(1);
  }
}
//------------------------------------------ main --------------------------------------------------

// machinery for replacing command line error by a macOS dialog box when running under Rosetta
#if defined(__APPLE__) && defined(__AVX__)
extern void DisplayErrorDialogBox(const char *title, const char *msg);
static const char *rosetta_error_msg = nullptr;
__attribute__((used, visibility("default"))) extern "C" void _mj_rosettaError(const char *msg)
{
  rosetta_error_msg = msg;
}
#endif

// user keyboard callback
void user_key_cb(GLFWwindow* window, int key, int scancode, int act, int mods) {
  if (act==GLFW_PRESS)
  {
    if (param::config.virtual_hoist && key == GLFW_KEY_H)
    {
      virtual_hoist_requested.store(true, std::memory_order_release);
      std::cout << "[HOIST] keyboard H request received\n";
    }
    if(param::config.enable_elastic_band == 1) {
      if (key==GLFW_KEY_9) {
        elastic_band.enable_ = !elastic_band.enable_;
      } else if (key==GLFW_KEY_7 || key==GLFW_KEY_UP) {
        elastic_band.length_ -= 0.1;
      } else if (key==GLFW_KEY_8 || key==GLFW_KEY_DOWN) {
        elastic_band.length_ += 0.1;
      }
    }
    if(key==GLFW_KEY_BACKSPACE) {
      int key_id = param::config.initial_keyframe.empty()
          ? -1
          : mj_name2id(m, mjOBJ_KEY, param::config.initial_keyframe.c_str());
      if (key_id >= 0) {
        mj_resetDataKeyframe(m, d, key_id);
      } else {
        mj_resetData(m, d);
      }
      mj_forward(m, d);
    }
  }
}

// run event loop
int main(int argc, char **argv)
{

  // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
  if (rosetta_error_msg)
  {
    DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
    std::exit(1);
  }
#endif

  // print version, check compatibility
  std::printf("MuJoCo version %s\n", mj_versionString());
  if (mjVERSION_HEADER != mj_version())
  {
    mju_error("Headers and library have different versions");
  }

  // scan for libraries in the plugin directory to load additional plugins
  scanPluginLibraries();

  mjvCamera cam;
  mjv_defaultCamera(&cam);

  mjvOption opt;
  mjv_defaultOption(&opt);

  mjvPerturb pert;
  mjv_defaultPerturb(&pert);

  // Load simulation configuration
  std::filesystem::path proj_dir = std::filesystem::path(getExecutableDir()).parent_path();
  param::config.load_from_yaml(proj_dir / "config.yaml");
  param::helper(argc, argv);
  if(param::config.robot_scene.is_relative()) {
    param::config.robot_scene = proj_dir.parent_path() / param::config.robot_scene;
  }

  // simulate object encapsulates the UI
  auto sim = std::make_unique<mj::Simulate>(
    std::make_unique<mj::GlfwAdapter>(),
    &cam, &opt, &pert, /* is_passive = */ false);

  std::thread unitree_thread(UnitreeSdk2BridgeThread, nullptr);

  // start physics thread
  std::thread physicsthreadhandle(&PhysicsThread, sim.get(), param::config.robot_scene.c_str());
  // start simulation UI loop (blocking call)
  glfwSetKeyCallback(static_cast<mj::GlfwAdapter*>(sim->platform_ui.get())->window_,user_key_cb);
  sim->RenderLoop();
  physicsthreadhandle.join();

  pthread_exit(NULL);
  return 0;
}
