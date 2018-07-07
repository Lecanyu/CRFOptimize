/**
 * @file visualizeInconsistentTracks.cpp
 * @brief visualize inconsistently labeled tracks
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/IO/Dataset.h"
#include "VideoParsing/IO/SequenceDataIO.h"
#include "VideoParsing/Utils/SemanticLabel.h"

#include <unordered_set>
#include <cstdlib>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/filesystem.hpp>

std::vector<vp::Track2D> getInconcsistentTracks(const std::vector<vp::Track2D>& tracks, const std::vector<cv::Mat>& label_images, int start_frame, int number_of_frames) {
  std::vector<vp::Track2D> inconsistent_tracks;
  const int last_frame = start_frame + number_of_frames;

  for (const vp::Track2D& track : tracks) {

    int track_start_frame = std::max(track.startFrameId(), start_frame);
    int track_end_frame = std::min(track.endFrameId(), last_frame);

    if ((track_start_frame < (last_frame-1)) && (track_end_frame > (start_frame + 1))) {

      std::unordered_set<int> labels;

      for (int frame_id = track_start_frame; frame_id < track_end_frame; ++frame_id) {
        Eigen::Vector2i loc = track.locationAtFrame(frame_id).array().round().cast<int>();
        cv::Vec3b color = label_images.at(frame_id - start_frame).at<cv::Vec3b>(loc.y(), loc.x());

        int label = static_cast<int>(vp::bgrToCamvid11(color));
        labels.insert(label);

        if(labels.size() > 1) {
          inconsistent_tracks.push_back(track);
          break;
        }

      }
    }
  }
  return inconsistent_tracks;
}

int main(int argc, char **argv) {

  if (argc < 3) {
    std::cout << "ERROR: parsing inputs\n";
    std::cout << "Usage: " << argv[0] << " config_file label_images_wild_card start_frame number_of_frames\n";
    return EXIT_FAILURE;
  }

  vp::Dataset dataset(argv[1]);

  const int start_frame = (argc > 3) ? std::stoi(argv[3]) : 0;
  const int number_of_frames = (argc > 4) ? std::stoi(argv[4]) : 801;

  std::vector<vp::Track2D> valid_tracks;
  {
    std::vector<vp::Track2D> tracks = dataset.loadTracks();
    std::cout << "Loaded " << tracks.size() << " tracks" << std::endl;

    std::copy_if(tracks.begin(), tracks.end(), std::back_inserter(valid_tracks),
                 vp::IsValidTrack(start_frame, number_of_frames));

    std::cout << "# Valid tracks = " << valid_tracks.size() << "\n";
  }

  std::cout << "Loading " << number_of_frames << " Frames [" << start_frame << ", "
            << start_frame + number_of_frames - 1 << "] .... " << std::flush;

  std::vector<cv::Mat> rgb_images = dataset.loadImageSequence(start_frame, number_of_frames);
  std::vector<cv::Mat> label_images = vp::loadImageSequence(argv[2], start_frame, number_of_frames);

  std::cout << "Done" << std::endl;

  std::vector<vp::Track2D> inconsistent_tracks = getInconcsistentTracks(valid_tracks, label_images,start_frame, number_of_frames );
  std::cout << "# of inconsistent tracks = " << inconsistent_tracks.size() << " ("
            << (inconsistent_tracks.size() * 100.0f) / valid_tracks.size() << "%)\n";

  cv::namedWindow("Inconsistent Tracks", cv::WINDOW_AUTOSIZE);

  bool paused = true;
  int step = 1;
  for(int i = 0;;) {
    const int frame_id = start_frame + i;
    cv::displayOverlay("Tracks", "Image #" + std::to_string(frame_id));

    for (const vp::Track2D& track : inconsistent_tracks) {
      if(track.isActive(frame_id)) {
        const Eigen::Vector2f& loc = track.locationAtFrame(frame_id);
        cv::Point point(std::round(loc.x()), std::round(loc.y()));
        cv::Vec3b color = label_images[i].at<cv::Vec3b>(point);
        cv::circle(rgb_images[i], point, 2, CV_RGB(color[2],color[1],color[0]), -1);
      }
    }

    cv::imshow("Inconsistent Tracks", rgb_images[i]);
    const int key = cv::waitKey(!paused);

    if (key == 27 || key == 81 || key == 113)  // Esc or Q or q
      break;
    else if (key == 123 || key == 125)  // Down or Left Arrow
      step = -1;
    else if (key == 124 || key == 126)  // Up or Right Arrow
      step = 1;
    else if (key == 'p' || key == 'P')
      paused = !paused;

    i += step;
    i = std::max(0, i);
    i = std::min(i, number_of_frames - 1);
  }

  return EXIT_SUCCESS;
}


