/**
 * @file exportOFTracks.cpp
 * @brief exportOFTracks
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/IO/TracksIO.h"
#include <iostream>

int main(int argc, char **argv) {
  using std::cout;

  if (argc != 3) {
    cout << "ERROR: parsing inputs\n";
    cout << "Usage: " << argv[0] << " input_tracks out_oftracks_file\n";
    return EXIT_FAILURE;
  }

  std::vector<vp::Track2D> tracks = vp::readTracks(argv[1]);
  vp::saveOFTracks(tracks, argv[2]);
  cout << "Saved " << tracks.size() << " tracks (OFTracks format) as " << argv[2] << "\n";

  return EXIT_SUCCESS;
}



