/**
 * @file SerializationHelper.h
 * @brief Boost Serialization helper functions
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_CORE_SERIALIZATION_HELPER_H_
#define VIDEOPARSING_CORE_SERIALIZATION_HELPER_H_

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>

#include <fstream>

namespace vp {

/**@brief Serializes data using Boost Serialization
 * @tparam Archive defaults to boost::archive::binary_oarchive
 * @tparam T data type to be serialized (should be automatically deduced)
 * @param[in] data object to serialize
 * @param[in] filename FileName to save to
 * @return true if success
 */
template <typename T, typename Archive>
bool serialize(const T& data, const std::string& filename) {
  std::ofstream ofs(filename.c_str());
  if (!ofs.is_open())
    return false;
  { // use scope to ensure archive goes out of scope before stream
    Archive oa(ofs);
    oa << data;
  }
  ofs.close();
  return true;
}

/**@brief DeSerializes data using Boost Serialization
 * @tparam Archive defaults to boost::archive::binary_iarchive
 * @tparam T data type to be De-Serialized (should be automatically deduced)
 * @param[out] data object to deSerialize
 * @param[in] filename FileName to load from
 * @return true if success
 */
template <typename T, typename Archive>
bool deSerialize(T& data, const std::string& filename) {
  std::ifstream ifs(filename.c_str());
  if (!ifs.is_open())
    return false;
  { // use scope to ensure archive goes out of scope before stream
    Archive ia(ifs);
    ia >> data;
  }
  ifs.close();
  return true;
}

/**@brief Serializes data using Boost Binary Archive
 * @tparam T data type to be serialized (should be automatically deduced)
 * @param[in] data object to serialize
 * @param[in] filename FileName to save to
 * @return true if success
 */
template <typename T>
bool serializeBinary(const T& data,  const std::string& filename) {
 return serialize<T, boost::archive::binary_oarchive>(data, filename);
}

/**@brief DeSerializes data using Boost Binary Archive
 * @tparam T data type to be De-Serialized (should be automatically deduced)
 * @param[out] data object to deSerialize
 * @param[in] filename FileName to load from
 * @return true if success
 */
template <typename T>
bool deSerializeBinary( T& data, const std::string& filename) {
  return deSerialize<T, boost::archive::binary_iarchive>(data, filename);
}



/**@brief Serializes data using Boost Text Archive
 * @tparam T data type to be serialized (should be automatically deduced)
 * @param[in] data object to serialize
 * @param[in] filename FileName to save to
 * @return true if success
 */
template <typename T>
bool serializeText(const T& data, const std::string& filename) {
  return serialize<T, boost::archive::text_oarchive>(data, filename);
}

/**@brief DeSerializes data using Boost Text Archive
 * @tparam T data type to be De-Serialized (should be automatically deduced)
 * @param[out] data object to deSerialize
 * @param[in] filename FileName to load from
 * @return true if success
 */
template <typename T>
bool deSerializeText(T& data, const std::string& filename) {
  return deSerialize<T, boost::archive::text_iarchive>(data, filename);
}

/**@brief Serializes data using Boost XML Archive
 * @tparam T data type to be serialized (should be automatically deduced)
 * @param[in] data object to serialize
 * @param[in] filename FileName to save to
 * @return true if success
 */
template <typename T>
bool serializeXml(const T& data, const std::string& filename) {
  std::ofstream ofs(filename.c_str());
  if (!ofs.is_open())
    return false;
  { // use scope to ensure archive goes out of scope before stream
    boost::archive::xml_oarchive oa(ofs);
    oa << BOOST_SERIALIZATION_NVP(data);
  }
  ofs.close();
  return true;
}

/**@brief DeSerializes data using Boost XML Archive
 * @tparam T data type to be De-Serialized (should be automatically deduced)
 * @param[out] data object to deSerialize
 * @param[in] filename FileName to load from
 * @return true if success
 */
template <typename T>
bool deSerializeXml(T& data, const std::string& filename) {
  std::ifstream ifs(filename.c_str());
  if (!ifs.is_open())
    return false;
  { // use scope to ensure archive goes out of scope before stream
    boost::archive::xml_iarchive ia(ifs);
    ia >> BOOST_SERIALIZATION_NVP(data);
  }
  ifs.close();
  return true;
}


} // end namespace vp

#endif // VIDEOPARSING_CORE_SERIALIZATION_HELPER_H_
