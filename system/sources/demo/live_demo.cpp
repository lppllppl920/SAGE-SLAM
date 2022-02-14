#include "live_demo.h"

#include <opencv2/imgproc/types_c.h>

namespace df
{
  /* ************************************************************************* */
  template <int CS>
  LiveDemo<CS>::LiveDemo(LiveDemoOptions opts) : opts_(opts), state_(INIT)
  {
    slam_system_ = nullptr;
    visualizer_ = nullptr;
    caminterface_ = nullptr;
    quit_ = false;

    live_timestamp_ = 0;

    dir_input = "";
    dir_crash = "";
  }

  /* ************************************************************************* */
  template <int CS>
  LiveDemo<CS>::~LiveDemo()
  {
    VLOG(3) << "[LiveDemo<CS>::~LiveDemo] deconstructor called";
    caminterface_.reset();
    slam_system_.reset();
    visualizer_.release();
  }

  LiveDemoOptions::InitType LiveDemoOptions::InitTypeTranslator(const std::string &s)
  {
    if (s == "ONEFRAME")
    {
      return InitType::ONEFRAME;
    }
    else if (s == "TWOFRAME")
    {
      return InitType::TWOFRAME;
    }
    else
    {
      LOG(FATAL) << "[LiveDemoOptions::InitTypeTranslator] Unknown init type: " << s;
    }

    return InitType::ONEFRAME;
  }

  /* ************************************************************************* */
  template <int CS>
  void LiveDemo<CS>::Run()
  {
    Init();
    if (opts_.enable_gui)
    {
      StartThreads();
    }

    ProcessingLoop();

    if (opts_.enable_gui)
    {
      JoinThreads();
    }
  }

  /* ************************************************************************* */
  template <int CS>
  void LiveDemo<CS>::Init()
  {
    // configure profiling
    EnableTiming(opts_.enable_timing);

    // create and setup camera
    // the category of camera interface is determined by the prefix of source_url
    caminterface_ = df::drivers::CameraInterfaceFactory::Get()->GetInterfaceFromUrl(opts_.source_url);

    // load camera intrinsics
    if (caminterface_->HasIntrinsics())
    {
      orig_cam_ = caminterface_->GetIntrinsics();
    }
    else
    {
      LOG(FATAL) << "[LiveDemo<CS>::Init] no camera intrinsics specified!";
    }
    // value range uint8 [0, 1]
    cv::Mat video_mask = caminterface_->GetMask();

    // for debugging -- create windows so that they dont pop up after init
    if (VLOG_IS_ON(3))
    {
      cv::startWindowThread();
      cv::namedWindow("reprojection errors", cv::WINDOW_AUTOSIZE);
      cv::namedWindow("se3 warping", cv::WINDOW_AUTOSIZE);
      cv::namedWindow("keyframes", cv::WINDOW_AUTOSIZE);
      cv::namedWindow("current frame", cv::WINDOW_AUTOSIZE);
      cv::namedWindow("camera tracking", cv::WINDOW_AUTOSIZE);
      cv::namedWindow("loop closures", cv::WINDOW_AUTOSIZE);

      cv::moveWindow("reprojection errors", 50, 0);
      cv::moveWindow("se3 warping", 1000, 0);
      cv::moveWindow("keyframes", 1700, 0);
      cv::moveWindow("camera tracking", 1700, 800);
      cv::moveWindow("current frame", 1700, 1400);
      cv::moveWindow("loop closures", 2200, 0);
    }

    // create & initialize slam system
    slam_system_ = std::make_unique<df::DeepFactors<float, CS>>();
    slam_system_->Init(orig_cam_, video_mask, opts_.df_opts);

    output_cam_ = orig_cam_.ComputeResizedCam(opts_.df_opts.net_output_size[1], opts_.df_opts.net_output_size[0]);
    if (!opts_.log_dir.empty())
    {
      CreateLogDirs();
    }

    // create & initialize visualizer
    df::VisualizerConfig vis_cfg;
    // category
    vis_cfg.num_visible_keyframes = opts_.num_visible_keyframes;
    vis_cfg.num_visible_frustums = opts_.num_visible_frustums;
    // bool
    vis_cfg.demo_mode = opts_.demo_mode;
    vis_cfg.frame_width = output_cam_.width();
    vis_cfg.frame_height = output_cam_.height();
    vis_cfg.record_video = opts_.record_video;
    vis_cfg.log_dir = opts_.log_dir;

    cv::resize(video_mask, video_mask, cv::Size2l(opts_.df_opts.net_output_size[1], opts_.df_opts.net_output_size[0]), 0, 0, CV_INTER_NN);
    video_mask.convertTo(video_mask, CV_32FC1);
    visualizer_ = std::make_unique<df::Visualizer>(vis_cfg, video_mask);

    // connect callbacks
    slam_system_->SetMapCallback(std::bind(&Visualizer::OnNewMap, &*visualizer_, std::placeholders::_1));
    slam_system_->SetPoseCallback(std::bind(&Visualizer::OnNewCameraPose, &*visualizer_, std::placeholders::_1));
    visualizer_->SetEventCallback(std::bind(&LiveDemo::HandleGuiEvent, this, std::placeholders::_1));
    slam_system_->SetStatsCallback(std::bind(&LiveDemo::StatsCallback, this, std::placeholders::_1));
  }

  /* ************************************************************************* */
  template <int CS>
  void LiveDemo<CS>::ProcessingLoop()
  {
    LOG(INFO) << "[LiveDemo<CS>::ProcessingLoop] Entering processing loop";

    // skipping specified number of frames once
    while (opts_.skip_frames > 0)
    {
      caminterface_->GrabFrames(live_timestamp_, &live_frame_);
      VLOG(3) << "[LiveDemo<CS>::ProcessingLoop] Skipping frame " << opts_.skip_frames;
      opts_.skip_frames--;
    }

    int frame_num = 0;

    if (opts_.init_on_start)
    {
      LOG(INFO) << "[LiveDemo<CS>::ProcessingLoop] Initializing system on the first frame";
      std::lock_guard<std::mutex> guard(slam_mutex_);
      caminterface_->GrabFrames(live_timestamp_, &live_frame_);
      VLOG(1) << "[LiveDemo<CS>::ProcessingLoop] Bootstrap frame " << live_timestamp_ << "(" << frame_num << ")";
      slam_system_->BootstrapOneFrame(live_timestamp_, live_frame_);
      state_ = DemoState::RUNNING;
      frame_num++;
    }

    int frame_count = 0;
    while (!quit_ && caminterface_->HasMore())
    {

      if (frame_count < opts_.frame_interval)
      {
        caminterface_->GrabFrames(live_timestamp_, &live_frame_);
        ++frame_count;
        continue;
      }
      else
      {
        caminterface_->GrabFrames(live_timestamp_, &live_frame_);
        frame_count = 0;
      }

      // // this variable will only be initilaized once for all following calls
      // static int retries = 4;

      // grab new image from camera
      // try
      // {

      //   retries = 4;
      // }
      // catch (std::exception &e)
      // {
      //   LOG(ERROR) << "[LiveDemo<CS>::ProcessingLoop] Grab frame error " << retries-- << ": " << e.what();
      //   if (retries <= 0)
      //   {
      //     LOG(FATAL) << "[LiveDemo<CS>::ProcessingLoop] Failed to grab too many times";
      //     quit_ = true;
      //   }
      //   continue;
      // }

      VLOG(1) << "[LiveDemo<CS>::ProcessingLoop] Process frame " << live_timestamp_ << "(" << frame_num << ")";

      if (opts_.frame_limit > 0 && frame_num > opts_.frame_limit)
      {
        VLOG(1) << "[LiveDemo<CS>::ProcessingLoop] Exiting because we've hit the frame limit (" << opts_.frame_limit << ")";
        break;
      }

      // record this image if thats requested
      if (opts_.record_input)
      {
        std::stringstream ss;
        ss << dir_input << "/" << std::setfill('0') << std::setw(4) << frame_num << ".png";
        cv::imwrite(ss.str(), live_frame_);
      }

      // give new frame to visualizer
      visualizer_->OnNewFrame(live_frame_);

      // feed the slam system
      if (state_ == RUNNING)
      {
        // The mutex seems to protect functions such as changing options in real-time and resetting etc.
        std::lock_guard<std::mutex> guard(slam_mutex_);
        tic("[LiveDemo<CS>::ProcessingLoop] ProcessFrame");
        slam_system_->ProcessFrame(live_timestamp_, live_frame_);

        toc("[LiveDemo<CS>::ProcessingLoop] ProcessFrame");
        if (opts_.enable_gui && VLOG_IS_ON(3))
        {
          cv::imshow("current frame", live_frame_);
        }
      }

      // for debugging -- show the stuff that slam system has imshowed
      if (opts_.enable_gui)
      {
        cv::waitKey(opts_.pause_step ? 0 : 1);
      }

      frame_num++;
    }

    if (!caminterface_->HasMore())
    {
      LOG(INFO) << "[LiveDemo<CS>::ProcessingLoop] No more frames to process!";
      slam_system_->JoinMappingThreads();
      while (!quit_ && !slam_system_->RefineMapping())
      {
      }
      slam_system_->JoinLoopThreads();
    }

    if (!opts_.log_dir.empty())
    {
      slam_system_->SaveResults(opts_.log_dir);
      // also save detailed info for debugging the system
      SaveInfo();
    }

    LOG(INFO) << "[LiveDemo<CS>::ProcessingLoop] Finished processing loop";

    if (opts_.quit_on_finish)
    {
      quit_ = true;
    }
  }

  /* ************************************************************************* */
  template <int CS>
  void LiveDemo<CS>::VisualizerLoop()
  {
    using namespace std;

    // create and init gui
    // NOTE: need to do this here so that opengl is initialized in the correct thread
    // NOTE(2): there's now a way how to fix that in pangolin
    visualizer_->Init(output_cam_, opts_.df_opts);

    const float loop_time = 1000.0f / 30.f;
    double elapsed_ms = 0;
    while (!quit_)
    {
      auto loop_start = chrono::steady_clock::now();

      // visualize
      visualizer_->HandleEvents();
      visualizer_->Draw(elapsed_ms);

      // regulate loop frequency to 60 hz
      auto loop_end = chrono::steady_clock::now();
      elapsed_ms = chrono::duration_cast<std::chrono::milliseconds>(loop_end - loop_start).count();

      int remaining_ms = loop_time - elapsed_ms;
      if (remaining_ms > 0)
      {
        std::this_thread::sleep_for(chrono::milliseconds(remaining_ms));
      }
    }
  }

  /* ************************************************************************* */
  template <int CS>
  void LiveDemo<CS>::StatsCallback(const DeepFactorsStatistics &stats)
  {
    DisplayStats disp_stats = {stats.relin_info};
    visualizer_->OnNewStats(disp_stats);
  }

  /* ************************************************************************* */
  template <int CS>
  void LiveDemo<CS>::HandleGuiEvent(GuiEvent evt)
  {
    switch (evt)
    {
    case GuiEvent::PARAM_CHANGE:
    {
      std::lock_guard<std::mutex> guard(slam_mutex_);
      opts_.df_opts = visualizer_->GetSlamOpts();
      slam_system_->SetOptions(opts_.df_opts);
      break;
    }
    case GuiEvent::EXIT:
      LOG(INFO) << "[LiveDemo<CS>::HandleGuiEvent] GUI requested exit";
      SaveInfo();
      quit_ = true;
      break;
    default:
      break;
    };
  }

  /* ************************************************************************* */
  template <int CS>
  void LiveDemo<CS>::StartThreads()
  {
    vis_thread_ = std::thread(std::bind(&LiveDemo::VisualizerLoop, this));
  }

  /* ************************************************************************* */
  template <int CS>
  void LiveDemo<CS>::JoinThreads()
  {
    vis_thread_.join();
    VLOG(2) << "[LiveDemo<CS>::JoinThreads] Threads joined";
  }

  /* ************************************************************************* */
  template <int CS>
  void LiveDemo<CS>::CreateLogDirs()
  {
    if (!opts_.log_dir.empty())
    {
      dir_input = opts_.log_dir + "/input";
    }
    else
    {
      dir_input = "input_" + df::GetTimeStamp();
    }

    if (opts_.record_input)
    {
      CreateDirIfNotExists(dir_input);

      std::ofstream file(dir_input + "/cam.txt");
      file << output_cam_.fx() << " " << output_cam_.fy() << " " << output_cam_.u0() << " " << output_cam_.v0() << " " << output_cam_.width() << " " << output_cam_.height();
      file.close();
    }
  }

  /* ************************************************************************* */
  template <int CS>
  void LiveDemo<CS>::SaveInfo()
  {
    // if we're logging somewhere, use that
    // if not, create a timestamped crash dir
    std::string out_dir = !opts_.log_dir.empty() ? opts_.log_dir : "crash_" + df::GetTimeStamp();
    df::CreateDirIfNotExists(out_dir);
    slam_system_->SaveInfo(out_dir);
    LOG(INFO) << "[LiveDemo<CS>::SaveInfo] Saved info to " << out_dir;
  }

  // explicit instantiation
  template class LiveDemo<DF_CODE_SIZE>;

} // namespace df
