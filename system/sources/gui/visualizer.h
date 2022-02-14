#ifndef DF_VISUALIZER_H_
#define DF_VISUALIZER_H_

#include <sophus/se3.hpp>
#include <pangolin/pangolin.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>
#include <shared_mutex>

#include "display_utils.h"
#include "timing.h"
#include "interp.h"
#include "keyframe_map.h"
#include "pinhole_camera.h"
#include "keyframe_renderer.h"
#include "deepfactors_options.h"

namespace cv
{
  class Mat;
}

namespace df
{

  enum GuiEvent
  {
    EXIT,
    PARAM_CHANGE,
  };

  struct VisualizerConfig
  {
    std::string win_title = "DeepFactors";
    int win_width = 1600;
    int win_height = 900;

    // colorscheme
    Eigen::Vector4f bg_color{0., 0., 0., 1.};
    Eigen::Vector4f cam_color{0.929, 0.443, 0.576, 1.};
    Eigen::Vector4f link_color{1.0, 0.0, 0.0, 1.};
    Eigen::Vector4f keyframe_color{0.58, 0.235, 1., 1.};
    Eigen::Vector4f frame_color{0.255, 0.063, 0.557, 1.};
    Eigen::Vector4f relin_color{0.992, 0.616, 0.322, 1.};

    int panel_width = 200;
    int frame_gap = 10;
    int frame_width = 200;
    int frame_height = 200;

    bool demo_mode = false;
    bool record_video = false;

    std::string log_dir = "";

    // number of keyframes to display
    int num_visible_keyframes = 5;
    // number of keyframe frustums to display
    int num_visible_frustums = 25;
  };

  struct DisplayStats
  {
    std::unordered_map<long, bool> relin_info;
  };

  class Visualizer
  {
  public:
    typedef std::function<void(GuiEvent)> EventCallback;
    typedef df::Map<float> MapT;
    typedef MapT::KeyframeT KeyframeT;
    typedef std::unordered_map<int, GuiEvent> KeyConfig;
    typedef MapT::KeyframeGraphT FrameGraphT;
    typedef FrameGraphT::LinkContainer LinkContainerT;
    typedef FrameGraphT::LinkT LinkT;
    typedef KeyframeRenderer::DisplayData KeyframeDisplayData;

    struct DisplayCacheItem
    {
      DisplayCacheItem(int width, int height)
          : data(width, height), active(false) {}

      KeyframeRenderer::DisplayData data;
      bool active;
    };

    Visualizer();
    Visualizer(const VisualizerConfig &cfg, const cv::Mat &video_mask);
    virtual ~Visualizer();

    void Init(const df::PinholeCamera<float> &cam, df::DeepFactorsOptions df_opts = df::DeepFactorsOptions{});
    bool ShouldQuit();

    void HandleEvents();
    void SyncOptions();
    void Draw(float delta_time);

    void OnNewFrame(const cv::Mat &frame);
    void OnNewCameraPose(const Sophus::SE3f &pose_wc);
    void OnNewMap(MapT::Ptr map);
    void OnNewStats(const DisplayStats &stats);

    void SetEventCallback(EventCallback cb);
    void SetKeymap(KeyConfig keycfg);

    void KeyCallback(int key);

    //  float GetHuberDelta() { return huber_delta_->Get(); }
    df::DeepFactorsOptions GetSlamOpts() const { return df_opts_; }

  private:
    void RenderKeyframes(float delta_time);
    void InitOpenGL();
    void BuildGui();
    void BuildPanel();
    void SaveResults();

    void EmitEvent(GuiEvent evt);

    // data to display
    df::PinholeCamera<float> cam_;
    Sophus::SE3f cam_pose_wc_;
    cv::Mat live_frame_;
    LinkContainerT keyframe_links_;
    LinkContainerT frame_links_;
    uint last_id_;
    bool initialized_ = false;

    // visualizer related things
    VisualizerConfig cfg_;
    KeyConfig keymap_;
    EventCallback event_callback_;
    df::KeyframeRenderer kfrenderer_;

    // GUI-related objects
    pangolin::View *map_viewport_;
    pangolin::View *frame_viewport_;
    pangolin::OpenGlRenderState *s_cam;
    pangolin::GlTexture *image_tex = nullptr;

    // gui/control options
    pangolin::Var<bool> *record_;
    pangolin::Var<bool> *phong_;
    pangolin::Var<bool> *freelook_;
    pangolin::Var<bool> *draw_traj_;
    pangolin::Var<bool> *draw_noisy_pix_;
    pangolin::Var<bool> *draw_all_frames_;
    pangolin::Var<float> *light_pos_x_;
    pangolin::Var<float> *light_pos_y_;
    pangolin::Var<float> *light_pos_z_;
    pangolin::Var<float> *trs_damping_;
    pangolin::Var<float> *rot_damping_;

    // mutexes
    std::mutex frame_mutex_;
    std::mutex campose_mutex_;
    std::mutex map_mutex_;
    std::mutex stats_mutex_;

    std::unordered_map<KeyframeT::IdType, DisplayCacheItem> display_data_;
    std::unordered_map<KeyframeT::IdType, Sophus::SE3f> keyframe_poses_;
    std::unordered_map<KeyframeT::IdType, Sophus::SE3f> frame_poses_;
    std::unordered_map<KeyframeT::IdType, bool> frame_mrg_;
    std::unordered_map<KeyframeT::IdType, bool> relin_info_;
    df::DeepFactorsOptions df_opts_;
    bool new_map_;

    // camera follow target
    Sophus::SE3f follow_pose_;

    // state for the smooth camera follow
    Eigen::Vector3f camera_vel_;
    Eigen::Vector4f camera_rot_vel_;

    // buffer for holding fading colors of keyframe frustums
    std::unordered_map<std::size_t, Eigen::Vector4f> color_buffer_;

    cv::Mat video_mask_;
  };

} // namespace df

#endif // DF_VISUALIZER_H_
