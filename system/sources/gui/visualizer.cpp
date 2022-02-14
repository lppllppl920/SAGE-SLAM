#include "visualizer.h"
#include "display_utils.h"

#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>

namespace df
{

  /* ************************************************************************* */
  Visualizer::Visualizer() : cfg_(VisualizerConfig()), map_viewport_(nullptr),
                             frame_viewport_(nullptr),
                             s_cam(nullptr),
                             image_tex(nullptr),
                             new_map_(true)
  {
    camera_rot_vel_.setZero();
    camera_vel_.setZero();
  }

  /* ************************************************************************* */
  Visualizer::Visualizer(const VisualizerConfig &cfg, const cv::Mat &video_mask)
      : cfg_(cfg),
        map_viewport_(nullptr),
        frame_viewport_(nullptr),
        s_cam(nullptr),
        image_tex(nullptr),
        new_map_(true)
  {
    // Erode the mask a bit to remove the boundary irregular depths
    cv::Mat erode_element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::erode(video_mask, video_mask_, erode_element);
    camera_rot_vel_.setZero();
    camera_vel_.setZero();
  }

  void GLAPIENTRY OglDebugCallback(GLenum source,
                                   GLenum type,
                                   GLuint id,
                                   GLenum severity,
                                   GLsizei length,
                                   const GLchar *message,
                                   const void *userParam)
  {
    VLOG(1) << "[OglDebugCallback] OpenGL Debug message: " << message;
    if (type == GL_DEBUG_TYPE_ERROR)
    {
      LOG(FATAL) << "[OglDebugCallback] OpenGL error: " << message;
    }
  }

  /* ************************************************************************* */
  Visualizer::~Visualizer()
  {
    VLOG(3) << "[Visualizer::~Visualizer] deconstructor called";

    delete image_tex;
    image_tex = nullptr;
    delete s_cam;
    s_cam = nullptr;
    delete frame_viewport_;
    frame_viewport_ = nullptr;
    delete map_viewport_;
    map_viewport_ = nullptr;
  }

  /* ************************************************************************* */
  void Visualizer::Init(const df::PinholeCamera<float> &cam, df::DeepFactorsOptions df_opts)
  {
    cam_ = cam;
    df_opts_ = df_opts;

    pangolin::CreateWindowAndBind(cfg_.win_title, cfg_.win_width, cfg_.win_height);

    InitOpenGL();
    BuildGui();

    kfrenderer_.Init(cam_);

    // here synchronize current GUI options with settings
    SyncOptions();

    // initialize camera position
    Eigen::Vector3f pos(0, 0, -1);
    Eigen::Vector3f rot(0, 0, 0);

    Sophus::SE3f cam_init;
    cam_init.translation() = pos;
    //from 3-rotation vector to 3x3-rotation matrix (matrix exponential operation)
    cam_init.so3() = Sophus::SO3f::exp(rot);
    s_cam->SetModelViewMatrix(cam_init.inverse().matrix());

    initialized_ = true;
  }

  /* ************************************************************************* */
  void Visualizer::SyncOptions()
  {
    kfrenderer_.SetPhong(phong_->Get());
    kfrenderer_.SetLightPos(light_pos_x_->Get(), light_pos_y_->Get(), light_pos_z_->Get());
  }

  /* ************************************************************************* */
  bool Visualizer::ShouldQuit()
  {
    // Returns true if user has requested to close OpenGL window.
    return pangolin::ShouldQuit();
  }

  /* ************************************************************************* */
  void Visualizer::HandleEvents()
  {
    if (ShouldQuit())
    {
      EmitEvent(EXIT);
    }

    if (pangolin::Pushed(*record_))
    {
      map_viewport_->RecordOnRender("ffmpeg:[fps=30,bps=8388608,unique_filename,flip=true]//" + cfg_.log_dir + "/screencap.avi");
    }

    if (pangolin::GuiVarHasChanged())
    {
      SyncOptions();
      EmitEvent(PARAM_CHANGE);
    }
  }

  /* ************************************************************************* */
  void Visualizer::Draw(float delta_time)
  {
    // Clear entire screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Activate 3d window
    map_viewport_->Activate(*s_cam);

    // Draw camera frustrum in the current pose
    if (draw_traj_->Get())
    {
      std::lock_guard<std::mutex> guard(campose_mutex_);
      glColor4f(cfg_.cam_color[0], cfg_.cam_color[1], cfg_.cam_color[2], cfg_.cam_color[3]);
      pangolin::glDrawFrustum<double>(cam_.InverseMatrix<double>(),
                                      cam_.width(), cam_.height(),
                                      cam_pose_wc_.matrix().cast<double>(), 0.05f);
    }

    if (!freelook_->Get())
    {
      std::lock_guard<std::mutex> guard(map_mutex_);

      // calculate current target pose
      Eigen::VectorXf off(6);
      off << 0, 0, -1, 0, 0, 0;
      Sophus::SE3f target = follow_pose_ * Sophus::SE3f::exp(off);

      // calculate smoothed camera pose
      Eigen::Matrix4f mv = Eigen::Matrix4f(s_cam->GetModelViewMatrix()).inverse();
      Eigen::Vector3f curr_pos = mv.topRightCorner<3, 1>();
      Eigen::Quaternionf curr_rot(mv.topLeftCorner<3, 3>());
      // here controls the behavior of damping viewpoint changing
      curr_pos = SmoothDamp(curr_pos, target.translation(), camera_vel_, delta_time / 1000.0f, trs_damping_->Get(), 10000000.f);
      curr_rot = QuatDamp(curr_rot, target.so3().unit_quaternion(), camera_rot_vel_, delta_time / 1000.f, rot_damping_->Get(), 1e-4);

      // set opengl cam
      Sophus::SE3f current_pose(curr_rot, curr_pos);
      s_cam->SetModelViewMatrix(current_pose.inverse().matrix());
    }
    else
    {
      // zero out velocities and just let the user control it
      camera_vel_.setZero();
      camera_rot_vel_.setZero();
    }

    // render all keyframes
    RenderKeyframes(delta_time);

    // upload live frame to texture
    if (!live_frame_.empty())
    {
      std::lock_guard<std::mutex> guard(frame_mutex_);
      if (!image_tex)
      {
        image_tex = new pangolin::GlTexture(live_frame_.cols, live_frame_.rows,
                                            GL_RGBA8, false, 0, GL_BGR,
                                            GL_UNSIGNED_BYTE);
      }

      image_tex->Upload(live_frame_.data, GL_BGR, GL_UNSIGNED_BYTE);
    }
    // Activate live frame window
    if (image_tex && !cfg_.demo_mode)
    {
      frame_viewport_->Activate();
      glColor3f(1.0, 1.0, 1.0);
      image_tex->RenderToViewportFlipY();
    }

    // Swap frames and Process Events
    pangolin::FinishFrame();
  }

  /* ************************************************************************* */
  void Visualizer::OnNewFrame(const cv::Mat &frame)
  {
    if (!initialized_)
    {
      return;
    }

    // frame_mutex_ is used to prevent the draw function from rendering an incomplete live_frame_ image
    std::lock_guard<std::mutex> guard(frame_mutex_);
    assert(frame.type() == CV_8UC3);
    cv::resize(frame, live_frame_, {int(frame.cols * (256.0 / frame.rows)), 256});
  }

  /* ************************************************************************* */
  void Visualizer::OnNewCameraPose(const Sophus::SE3f &pose_wc)
  {
    std::lock_guard<std::mutex> guard(campose_mutex_);
    cam_pose_wc_ = pose_wc;
  }

  /* ************************************************************************* */
  void Visualizer::OnNewMap(MapT::Ptr map)
  {
    if (!initialized_)
    {
      return;
    }

    std::lock_guard<std::mutex> guard(map_mutex_);
    keyframe_links_ = map->keyframes.GetLinks();

    // copy frame -> keyframe links
    frame_links_.clear();
    for (auto &link : map->frames.GetLinks())
      frame_links_.push_back(link);

    // get the frame poses for display
    frame_poses_.clear();
    for (auto &id : map->frames.Ids())
    {
      frame_poses_[id] = map->frames.Get(id)->pose_wk;
      frame_mrg_[id] = false; //map->frames.Get(id)->marginalized;
    }

    // mark all cache items as inactive
    for (auto &kv : display_data_)
      kv.second.active = false;

    last_id_ = map->keyframes.LastId();

    // copy all up-to-date keyframe poses
    keyframe_poses_.clear();
    for (auto &id : map->keyframes.Ids())
      keyframe_poses_[id] = map->keyframes.Get(id)->pose_wk;

    // Copy only last N frames
    int max = static_cast<int>(last_id_) - cfg_.num_visible_keyframes;
    for (int id = last_id_; id > 0; --id)
    {
      // If now drawing all frames, only display the keyframes within the specified range
      if (!draw_all_frames_->Get() && id <= max)
      {
        break;
      }

      // check if kf already has an KeyframeDisplayData object
      if (display_data_.find(id) == display_data_.end())
      {
        display_data_.emplace(id, DisplayCacheItem{static_cast<int>(cam_.width()), static_cast<int>(cam_.height())});
      }

      // fill the data using keyframe
      auto kf = map->keyframes.Get(id);
      auto &item = display_data_.at(id);
      auto &data = item.data;
      // resize the color_img to be the same size as dpt_map etc.
      cv::resize(kf->color_img, data.color_img, cv::Size2l(static_cast<long>(cam_.width()), static_cast<long>(cam_.height())));
      {
        std::shared_lock<std::shared_mutex> lock(kf->mutex);
        data.pose_wk = kf->pose_wk;
        data.dpt = kf->dpt_map.to(torch::kCPU).contiguous().clone();
      }
      
      data.vld = video_mask_.mul(Tensor2Mat(data.dpt > 1.0e-2));
      // VLOG(2) << "video mask for visualization: " << data.vld.size();
      data.dpt = torch::clamp_min(data.dpt, 1.0e-2);

      item.active = true;
    }

    if (last_id_ > 0)
    {
      follow_pose_ = map->keyframes.Get(last_id_)->pose_wk;
    }
    new_map_ = true;
  }

  /* ************************************************************************* */
  void Visualizer::OnNewStats(const DisplayStats &stats)
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    relin_info_ = stats.relin_info;
  }

  /* ************************************************************************* */
  void Visualizer::SetEventCallback(EventCallback cb)
  {
    event_callback_ = cb;
  }

  /* ************************************************************************* */
  void Visualizer::SetKeymap(KeyConfig keycfg)
  {
    keymap_ = keycfg;
    // register callback for required keys
    for (auto &map : keymap_)
      pangolin::RegisterKeyPressCallback(map.first, std::bind(&Visualizer::KeyCallback, this, std::placeholders::_1));
  }

  /* ************************************************************************* */
  void Visualizer::KeyCallback(int key)
  {
    if (keymap_.find(key) != keymap_.end())
    {
      EmitEvent(keymap_[key]);
    }
  }

  /* ************************************************************************* */
  void Visualizer::RenderKeyframes(float delta_time)
  {
    std::lock_guard<std::mutex> guard(map_mutex_);

    pangolin::OpenGlMatrix vp = s_cam->GetProjectionModelViewMatrix();

    std::unordered_map<KeyframeT::IdType, bool> relin_info;
    {
      std::lock_guard<std::mutex> lock(stats_mutex_);
      relin_info = std::unordered_map<KeyframeT::IdType, bool>(relin_info_);
    }

    for (auto &kv : display_data_)
    {
      // Not drawing inactive keyframes
      if (!kv.second.active)
      {
        continue;
      }
      {
        auto &data = kv.second.data;
        kfrenderer_.RenderKeyframe(vp, data);
      }
    }

    if (!draw_traj_->Get())
    {
      return;
    }

    int last_drawn_depth_id = static_cast<int>(last_id_) - cfg_.num_visible_keyframes;
    int last_drawn_frustum_id = static_cast<int>(last_id_) - cfg_.num_visible_frustums;
    // render keyframe frustums
    for (auto &kv : keyframe_poses_)
    {
      auto id = kv.first;
      auto pose_wk = kv.second;

      if (draw_all_frames_->Get())
      {
        color_buffer_[id] = cfg_.keyframe_color;
      }
      else
      {
        // set the appropriate color
        if (static_cast<int>(id) >= last_drawn_depth_id) // if the depth is displayed
        {
          color_buffer_[id] = cfg_.keyframe_color;
        }
        else if (static_cast<int>(id) >= last_drawn_frustum_id) // if the frustum is to be displayed
        {
          color_buffer_[id] = cfg_.frame_color;
        }
        else // fade out
        {
          color_buffer_[id][3] = 0.f;
        }
      }

      if (relin_info[id])
      {
        color_buffer_[id] = cfg_.relin_color;
        relin_info[id] = false;
      }

      glLineWidth(2.);
      Eigen::Vector4f color = color_buffer_[id];
      if (color[3] < 0.01)
      {
        continue;
      }
      glColor4f(color[0], color[1], color[2], color[3]);
      pangolin::glDrawFrustum<double>(cam_.InverseMatrix<double>(),
                                      cam_.width(), cam_.height(),
                                      pose_wk.matrix().cast<double>(), 0.05f);
    }

    // render links between keyframes
    for (auto &link : keyframe_links_)
    {
      auto pos1 = keyframe_poses_[link.first].translation();
      auto pos2 = keyframe_poses_[link.second].translation();

      auto col1 = color_buffer_[link.first];
      auto col2 = color_buffer_[link.second];

      if (col1[3] < 0.01 || col2[3] < 0.01)
      {
        continue;
      }

      glLineWidth(1.);
      GLfloat vertices[6] = {pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2]};
      GLfloat colors[8] = {col1[0], col1[1], col1[2], col1[3], col2[0], col2[1], col2[2], col2[3]};
      pangolin::glDrawColoredVertices<float, float>(2, vertices, colors, GL_LINES, 3, 4);
    }
  }

  /* ************************************************************************* */
  void Visualizer::InitOpenGL()
  {
    glEnable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);

    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

    glClearColor(cfg_.bg_color[0], cfg_.bg_color[1], cfg_.bg_color[2], cfg_.bg_color[3]);

    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(OglDebugCallback, nullptr);
  }

  /* ************************************************************************* */
  void Visualizer::BuildGui()
  {
    // Define Camera Render Object (for view / scene browsing)
    df::PinholeCamera<float> cam(cam_);
    cam.ResizeViewport(cfg_.win_width, cfg_.win_height);
    s_cam = new pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrixRDF_TopLeft(cam.width(), cam.height(), cam.fx(), cam.fy(), cam.u0(), cam.v0(), 0.01, 100),
        pangolin::ModelViewLookAtRDF(0, 0, -1, 0, 0, 0, 0, -1, 0));

    // Create viewport for map visualization
    pangolin::Attach left_bound = cfg_.demo_mode ? 0.f : pangolin::Attach::Pix(cfg_.panel_width);
    map_viewport_ = &pangolin::CreateDisplay()
                         .SetBounds(0.0, 1.0, left_bound, 1.0, -640.0f / 480.0f)
                         .SetHandler(new pangolin::Handler3D(*s_cam));

    // Create viewport for live frame visualization
    if (!cfg_.demo_mode)
    {
      frame_viewport_ = &pangolin::CreateDisplay()
                             .SetLock(pangolin::LockLeft, pangolin::LockTop)
                             .SetBounds(pangolin::Attach::Pix(cfg_.frame_gap),
                                        pangolin::Attach::Pix(cfg_.frame_gap + 256),
                                        pangolin::Attach::Pix(cfg_.panel_width + cfg_.frame_gap),
                                        pangolin::Attach::Pix(cfg_.panel_width + cfg_.frame_gap + int(256.0 / cfg_.frame_height * cfg_.frame_width)));
    }

    // Create and populate left panel with buttons
    BuildPanel();
    // shortcuts to toggle several functions
    pangolin::RegisterKeyPressCallback('f', pangolin::ToggleVarFunctor("ui.FreeLook"));
    pangolin::RegisterKeyPressCallback('t', pangolin::ToggleVarFunctor("ui.DrawTrajectory"));
    pangolin::RegisterKeyPressCallback('h', pangolin::ToggleVarFunctor("ui.DrawAllFrames"));
  }

  /* ************************************************************************* */
  void Visualizer::BuildPanel()
  {
    if (!cfg_.demo_mode)
    {
      pangolin::CreatePanel("ui")
          .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(cfg_.panel_width));
    }

    // [default, toggle]
    record_ = new pangolin::Var<bool>("ui.Record", cfg_.record_video, false);
    phong_ = new pangolin::Var<bool>("ui.Phong", false, true);
    freelook_ = new pangolin::Var<bool>("ui.FreeLook", false, true);
    draw_traj_ = new pangolin::Var<bool>("ui.DrawTrajectory", true, true);
    draw_all_frames_ = new pangolin::Var<bool>("ui.DrawAllFrames", true, true);

    // [default, min, max]
    light_pos_x_ = new pangolin::Var<float>("ui.LightPosX", 0.0, -3.0, 3.0);
    light_pos_y_ = new pangolin::Var<float>("ui.LightPosY", 0.0, -3.0, 3.0);
    light_pos_z_ = new pangolin::Var<float>("ui.LightPosZ", 0.0, -3.0, 3.0);
    trs_damping_ = new pangolin::Var<float>("ui.TransDamping", 0.5, 0.01, 1.5);
    rot_damping_ = new pangolin::Var<float>("ui.RotDamping", 2.0, 0.01, 20.0);
  }

  /* ************************************************************************* */
  void Visualizer::SaveResults()
  {
  }

  /* ************************************************************************* */
  void Visualizer::EmitEvent(GuiEvent evt)
  {
    event_callback_(evt);
  }

} // namespace df
