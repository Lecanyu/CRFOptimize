/**
 * @file ConcurrentHashTable.h
 * @brief ConcurrentHashTable
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_DENSEGM_CONCURRENTHASHTABLE_H_
#define VIDEOPARSING_DENSEGM_CONCURRENTHASHTABLE_H_

#include <vector>
#include <mutex>

namespace vp {

template <class KeyType>
class ConcurrentHashTable {
  static_assert(KeyType::SizeAtCompileTime != Eigen::Dynamic, "Requires fixed size key type");
  typedef std::size_t size_type;
  typedef typename KeyType::Index KeyIndex;

 public:
  explicit ConcurrentHashTable(size_type n_elements)
      : filled_(0),
        capacity_(2 * n_elements),
        keys_((capacity_ / 2 + 10)),
        table_(2 * n_elements, -1) {
  }
  size_type size() const {
    return filled_;
  }
  void reset() {
    filled_ = 0;
    std::fill(table_.begin(), table_.end(), -1);
  }

  int find(const KeyType& key) const {
    // Get the hash value
    size_t h = hash(key) % capacity_;
    // Find the element with he right key, using linear probing
    while (1) {
      int e = table_[h];
      if (e == -1) {
          return -1;
      }
      // Check if the current key is The One
      if(key == keys_[e])
        return e;

      // Continue searching
      h++;
      if (h == capacity_)
        h = 0;
    }
  }

  int insert(const KeyType& key) {
    if (2 * filled_ >= capacity_)
      grow();
    // Get the hash value
    size_t h = hash(key) % capacity_;
    // Find the element with he right key, using linear probing
    while (1) {
      int e = table_[h];
      if (e == -1) {
        std::lock_guard<std::mutex> lock(mutex_);

        // double buffer
        e = table_[h];
        if(e == -1) {
          // Insert a new key and return the new id
          keys_[filled_] = key;
          return table_[h] = filled_++;
        }
      }

      // Check if the current key is The One
      if (key == keys_[e])
        return e;

      // Continue searching
      ++h;
      if (h == capacity_)
        h = 0;
    }
  }
  const KeyType& getKey(int i) const {
    return keys_[i];
  }

 protected:
  size_type filled_, capacity_;
  std::vector<KeyType> keys_;
  std::vector<int> table_;
  std::mutex mutex_;

  void grow() {
    throw std::runtime_error("Growing not handled");
    // Create the new memory and copy the values in
    size_type old_capacity = capacity_;
    capacity_ *= 2;
    std::vector<KeyType> old_keys((old_capacity + 10));
    std::copy(keys_.begin(), keys_.end(), old_keys.begin());
    std::vector<int> old_table(capacity_, -1);

    // Swap the memory
    table_.swap(old_table);
    keys_.swap(old_keys);

    // Reinsert each element
    for (size_type i = 0; i < old_capacity; ++i)
      if (old_table[i] >= 0) {
        int e = old_table[i];
        size_type h = hash(getKey(e)) % capacity_;
        for (; table_[h] >= 0; h = h < capacity_ - 1 ? h + 1 : 0);
        table_[h] = e;
      }
  }

  size_type hash(const KeyType& k) const {
    size_type r = 0;
    for (KeyIndex i = 0; i < KeyType::SizeAtCompileTime; ++i) {
      r += k[i];
      r *= 1664525;
    }
    return r;
  }
};

}  // namespace vp

#endif /* VIDEOPARSING_DENSEGM_CONCURRENTHASHTABLE_H_ */
