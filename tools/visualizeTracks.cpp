/**
 * @file visualizeTracks.cpp
 * @brief visualizeTracks
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/IO/Dataset.h"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

cv::Vec3b hsvToBgr8U (float h, float s, float v) {
  float r=0;
  float g=0;
  float b=0;

  float c  = v*s;
  float h2 = 6.0*h;
  float x  = c*(1.0-fabs(fmod(h2,2.0)-1.0));
  if (0<=h2&&h2<1)       { r = c; g = x; b = 0; }
  else if (1<=h2&&h2<2)  { r = x; g = c; b = 0; }
  else if (2<=h2&&h2<3)  { r = 0; g = c; b = x; }
  else if (3<=h2&&h2<4)  { r = 0; g = x; b = c; }
  else if (4<=h2&&h2<5)  { r = x; g = 0; b = c; }
  else if (5<=h2&&h2<=6) { r = c; g = 0; b = x; }
  else if (h2>6) { r = 1; g = 0; b = 0; }
  else if (h2<0) { r = 0; g = 1; b = 0; }

  return cv::Vec3b(255. * b, 255. * g, 255. * r);
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cout << "ERROR: dataset config file not provided\n";
    std::cout << "Usage: " << argv[0] << " config_file start_frame number_of_frames\n";
    return EXIT_FAILURE;
  }

  vp::Dataset dataset(argv[1]);
  dataset.print();
  const int start_frame = std::stoi(argv[2]);
  const int number_of_frames = std::stoi(argv[3]);

  std::cout << "Loading Tracks .. " << std::flush;
  std::vector<vp::Track2D> tracks = dataset.loadTracks(start_frame, number_of_frames);
  std::cout << " Done. Found " << tracks.size() << " tracks." << std::endl;

  cv::namedWindow("Tracks", cv::WINDOW_AUTOSIZE);

  bool paused = true;
  int step = 1;
  for(int i = 0;;) {
    const int frame_id = start_frame + i;
    cv::displayOverlay("Tracks", "Image #" + std::to_string(frame_id));

    cv::Mat image = dataset.loadImage(frame_id);

    if(image.empty()) {
      std::cout << "Empty Image at frameId: " << frame_id << ". So quitting..\n";
      break;
    }

    for (const vp::Track2D& track : tracks) {
      if (track.isActive(frame_id)) {
        int track_age = frame_id - track.startFrameId();

        const Eigen::Vector2i loci = track.locations[track_age].array().round().cast<int>();

        if (loci.x() < 0 || loci.y() < 0 || loci.x() >= image.cols || loci.y() >= image.rows) {
          std::cout << "Bad Location " << loci.transpose() << "\n";
        }

        float h = track_age * (10.0f / number_of_frames);

        cv::Vec3b color = hsvToBgr8U(h, 1, 1);

        image.at<cv::Vec3b>(loci.y(), loci.x()) = color;
      }
    }

    cv::imshow("Tracks", image);
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
