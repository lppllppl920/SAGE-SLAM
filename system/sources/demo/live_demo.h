#ifndef DF_LIVE_DEMO_H_
#define DF_LIVE_DEMO_H_

#include <memory>
#include <vector>
#include <thread>
#include <torch/torch.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <functional>
#include <boost/filesystem.hpp>

#include "tum_io.h"
#include "logutils.h"
#include "visualizer.h"
#include "timing.h"
#include "camera_interface_factory.h"
#include "deepfactors.h"
#include "visualizer.h"

namespace df
{

  // forward declarations
  class Visualizer;
  enum GuiEvent;
  namespace drivers
  {
    class CameraInterface;
  }

  struct LiveDemoOptions
  {
    enum InitType
    {
      ONEFRAME,
      TWOFRAME
    };

    InitType init_type = TWOFRAME;
    df::DeepFactorsOptions df_opts;
    std::string source_url;
    bool record_input = false;
    std::string record_path;
    bool init_on_start;
    bool pause_step;
    std::string log_dir;
    bool enable_timing;
    bool quit_on_finish;
    bool enable_gui;
    int frame_limit = 0;
    int skip_frames;
    bool demo_mode;
    bool record_video;
    int num_visible_keyframes;
    int num_visible_frustums;
    int frame_interval;

    static InitType InitTypeTranslator(const std::string &s);
  };

  /*
 Class representing the live demo program
 Handles system things:
    Starting/stopping threads
    Fetching new frame from camera
    Feeding the slam system
*/
  template <int CS>
  class LiveDemo
  {
  public:
    enum DemoState
    {
      RUNNING,
      PAUSED,
      INIT
    };

    typedef df::DeepFactors<float, CS> SlamT;

    LiveDemo(LiveDemoOptions opts);
    ~LiveDemo();

    void Run();

  private:
    void Init();
    void ProcessingLoop();
    void VisualizerLoop();
    void StatsCallback(const DeepFactorsStatistics &stats);
    void HandleGuiEvent(GuiEvent evt);

    void StartThreads();
    void JoinThreads();

    void CreateLogDirs();
    void SaveInfo();

    df::LiveDemoOptions opts_;
    std::atomic<DemoState> state_;

    std::unique_ptr<SlamT> slam_system_;
    std::unique_ptr<df::Visualizer> visualizer_;
    std::unique_ptr<df::drivers::CameraInterface> caminterface_;

    bool quit_;

    double live_timestamp_;
    cv::Mat live_frame_;

    // directories that demo is logging to
    std::string dir_input;
    std::string dir_crash;

    std::thread vis_thread_;
    std::mutex slam_mutex_;
    // connected camera intrinsics
    df::PinholeCamera<float> output_cam_;
    df::PinholeCamera<float> orig_cam_;


  };

} // namespace df

#endif // DF_LIVE_DEMO_H_
