/**
 * @file SemanticLabel.cpp
 * @brief SemanticLabel
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/Utils/SemanticLabel.h"

namespace vp {

Camvid11 mergeCamvid32ToCamvid11(Camvid32 label) {
  Camvid11 meged_label = Camvid11::VOID;
  switch (label) {
    case Camvid32::VOID: meged_label = Camvid11::VOID; break;
    case Camvid32::BUILDING: meged_label = Camvid11::BUILDING; break;
    case Camvid32::TREE: meged_label = Camvid11::TREE; break;
    case Camvid32::SKY: meged_label = Camvid11::SKY; break;
    case Camvid32::CAR: meged_label = Camvid11::CAR; break;
    case Camvid32::SIGNSYMBOL: meged_label = Camvid11::SIGN_SYMBOL; break;
    case Camvid32::ROAD: meged_label = Camvid11::ROAD; break;
    case Camvid32::PEDESTRIAN: meged_label = Camvid11::PEDESTRIAN; break;
    case Camvid32::FENCE: meged_label = Camvid11::FENCE; break;
    case Camvid32::COLUMN_POLE: meged_label = Camvid11::COLUMN_POLE; break;
    case Camvid32::SIDEWALK: meged_label = Camvid11::SIDEWALK; break;
    case Camvid32::BICYCLIST: meged_label = Camvid11::BICYCLIST; break;

    // merged labels
    case Camvid32::VEGETATIONMISC: meged_label = Camvid11::TREE; break;
    case Camvid32::ROADSHOULDER: meged_label = Camvid11::ROAD; break;
    case Camvid32::LANEMKGSDRIV: meged_label = Camvid11::ROAD; break;
    case Camvid32::LANEMKGSNONDRIV: meged_label = Camvid11::ROAD; break;
    case Camvid32::PARKINGBLOCK: meged_label = Camvid11::SIDEWALK; break;
    case Camvid32::TRAFFICLIGHT: meged_label = Camvid11::SIGN_SYMBOL; break;
    case Camvid32::MOTORCYCLESCOOTER: meged_label = Camvid11::BICYCLIST; break;
    case Camvid32::TRUCK_BUS: meged_label = Camvid11::CAR; break;
    case Camvid32::SUVPICKUPTRUCK: meged_label = Camvid11::CAR; break;
    case Camvid32::MISC_TEXT: meged_label = Camvid11::SIGN_SYMBOL; break;
    case Camvid32::WALL: meged_label = Camvid11::BUILDING; break;

    // Ignored labels
    case Camvid32::ARCHWAY: meged_label = Camvid11::VOID; break;
    case Camvid32::BRIDGE: meged_label = Camvid11::VOID; break;
    case Camvid32::TUNNEL: meged_label = Camvid11::VOID; break;
    case Camvid32::ANIMAL: meged_label = Camvid11::VOID; break;
    case Camvid32::CARTLUGGAGEPRAM: meged_label = Camvid11::VOID; break;
    case Camvid32::CHILD: meged_label = Camvid11::VOID; break;
    case Camvid32::OTHERMOVING: meged_label = Camvid11::VOID; break;
    case Camvid32::TRAFFICCONE: meged_label = Camvid11::VOID; break;
    case Camvid32::TRAIN: meged_label = Camvid11::VOID; break;

  }
  return meged_label;
}

Camvid32 bgrToCamvid32(const cv::Vec3b& bgr) {
  Camvid32 label = Camvid32::VOID;

  if(bgr == cv::Vec3b(0,0,0)) label = Camvid32::VOID;
  else if(bgr == cv::Vec3b(128,64,128)) label = Camvid32::ROAD;
  else if(bgr == cv::Vec3b(0,0,128)) label = Camvid32::BUILDING;
  else if(bgr == cv::Vec3b(128,128,128)) label = Camvid32::SKY;
  else if(bgr == cv::Vec3b(0,128,128)) label = Camvid32::TREE;
  else if(bgr == cv::Vec3b(192,0,0)) label = Camvid32::SIDEWALK;
  else if(bgr == cv::Vec3b(128,0,64)) label = Camvid32::CAR;
  else if(bgr == cv::Vec3b(128,64,64)) label = Camvid32::FENCE;
  else if(bgr == cv::Vec3b(128,192,192)) label = Camvid32::COLUMN_POLE;
  else if(bgr == cv::Vec3b(0,64,64)) label = Camvid32::PEDESTRIAN;
  else if(bgr == cv::Vec3b(128,128,192)) label = Camvid32::SIGNSYMBOL;
  else if(bgr == cv::Vec3b(192,128,0)) label = Camvid32::BICYCLIST;
  else if(bgr == cv::Vec3b(0,192,64)) label = Camvid32::WALL;
  else if(bgr == cv::Vec3b(192,128,128)) label = Camvid32::ROADSHOULDER;
  else if(bgr == cv::Vec3b(0,192,192)) label = Camvid32::VEGETATIONMISC;
  else if(bgr == cv::Vec3b(192,0,192)) label = Camvid32::MOTORCYCLESCOOTER;
  else if(bgr == cv::Vec3b(192,0,128)) label = Camvid32::LANEMKGSDRIV;
  else if(bgr == cv::Vec3b(64,0,192)) label = Camvid32::LANEMKGSNONDRIV;
  else if(bgr == cv::Vec3b(128,192,64)) label = Camvid32::PARKINGBLOCK;
  else if(bgr == cv::Vec3b(192,128,64)) label = Camvid32::SUVPICKUPTRUCK;
  else if(bgr == cv::Vec3b(128,0,192)) label = Camvid32::ARCHWAY;
  else if(bgr == cv::Vec3b(64,128,0)) label = Camvid32::BRIDGE;
  else if(bgr == cv::Vec3b(192,0,64)) label = Camvid32::CARTLUGGAGEPRAM;
  else if(bgr == cv::Vec3b(64,128,192)) label = Camvid32::CHILD;
  else if(bgr == cv::Vec3b(64,128,128)) label = Camvid32::MISC_TEXT;
  else if(bgr == cv::Vec3b(64,64,128)) label = Camvid32::OTHERMOVING;
  else if(bgr == cv::Vec3b(64,0,0)) label = Camvid32::TRAFFICCONE;
  else if(bgr == cv::Vec3b(64,64,0)) label = Camvid32::TRAFFICLIGHT;
  else if(bgr == cv::Vec3b(128,64,192)) label = Camvid32::TRAIN;
  else if(bgr == cv::Vec3b(192,128,192)) label = Camvid32::TRUCK_BUS;
  else if(bgr == cv::Vec3b(64,0,64)) label = Camvid32::TUNNEL;
  else if(bgr == cv::Vec3b(64,128,64)) label = Camvid32::ANIMAL;
  else throw std::invalid_argument("invalid Color");

  return label;
}

Camvid11 bgrToCamvid11(const cv::Vec3b& bgr) {
  Camvid11 label = Camvid11::VOID;

  if(bgr == cv::Vec3b(0,0,0)) label = Camvid11::VOID;
  else if(bgr == cv::Vec3b(128,64,128)) label = Camvid11::ROAD;
  else if(bgr == cv::Vec3b(0,0,128)) label = Camvid11::BUILDING;
  else if(bgr == cv::Vec3b(128,128,128)) label = Camvid11::SKY;
  else if(bgr == cv::Vec3b(0,128,128)) label = Camvid11::TREE;
  else if(bgr == cv::Vec3b(192,0,0)) label = Camvid11::SIDEWALK;
  else if(bgr == cv::Vec3b(128,0,64)) label = Camvid11::CAR;
  else if(bgr == cv::Vec3b(128,64,64)) label = Camvid11::FENCE;
  else if(bgr == cv::Vec3b(128,192,192)) label = Camvid11::COLUMN_POLE;
  else if(bgr == cv::Vec3b(0,64,64)) label = Camvid11::PEDESTRIAN;
  else if(bgr == cv::Vec3b(128,128,192)) label = Camvid11::SIGN_SYMBOL;
  else if(bgr == cv::Vec3b(192,128,0)) label = Camvid11::BICYCLIST;
  else throw std::invalid_argument("invalid Color");

  return label;
}

CityScape19 bgrToCityScape19(const cv::Vec3b& bgr) {
  CityScape19 label = CityScape19::IGNORE;

  if(bgr == cv::Vec3b(128, 64,128)) label = CityScape19::ROAD;
  else if(bgr == cv::Vec3b(232, 35, 244)) label = CityScape19::SIDEWALK;
  else if(bgr == cv::Vec3b(70, 70, 70)) label = CityScape19::BUILDING;
  else if(bgr == cv::Vec3b(156, 102, 102)) label = CityScape19::WALL;
  else if(bgr == cv::Vec3b(153, 153, 190)) label = CityScape19::FENCE;
  else if(bgr == cv::Vec3b(153, 153, 153)) label = CityScape19::POLE;
  else if(bgr == cv::Vec3b(30, 170, 250)) label = CityScape19::TRAFFIC_LIGHT;
  else if(bgr == cv::Vec3b(0, 220, 220)) label = CityScape19::TRAFFIC_SIGN;
  else if(bgr == cv::Vec3b(35, 142, 107)) label = CityScape19::VEGETATION;
  else if(bgr == cv::Vec3b(152,251,152)) label = CityScape19::TERRAIN;
  else if(bgr == cv::Vec3b(180,130,70)) label = CityScape19::SKY;
  else if(bgr == cv::Vec3b(60, 20, 220)) label = CityScape19::PERSON;
  else if(bgr == cv::Vec3b(0, 0, 255)) label = CityScape19::RIDER;
  else if(bgr == cv::Vec3b(142, 0, 0)) label = CityScape19::CAR;
  else if(bgr == cv::Vec3b(70, 0, 0)) label = CityScape19::TRUCK;
  else if(bgr == cv::Vec3b(100, 60, 0)) label = CityScape19::BUS;
  else if(bgr == cv::Vec3b(100, 80, 0)) label = CityScape19::TRAIN;
  else if(bgr == cv::Vec3b(230, 0, 0)) label = CityScape19::MOTORCYCLE;
  else if(bgr == cv::Vec3b(32, 11, 119)) label = CityScape19::BICYCLE;

  return label;
}

}  // namespace vp

