/**
 * serialize_utils.hpp — Binary I/O helpers for engine persistence.
 *
 * All functions assume little-endian byte order (x86/x64 native).
 */

#pragma once

#include <cstdint>
#include <istream>
#include <ostream>
#include <string>

namespace trifecta::io {

inline void write_u8 (std::ostream& os, uint8_t  v) { os.write(reinterpret_cast<const char*>(&v), 1); }
inline void write_u32(std::ostream& os, uint32_t v) { os.write(reinterpret_cast<const char*>(&v), 4); }
inline void write_u64(std::ostream& os, uint64_t v) { os.write(reinterpret_cast<const char*>(&v), 8); }
inline void write_i32(std::ostream& os, int32_t  v) { os.write(reinterpret_cast<const char*>(&v), 4); }
inline void write_f32(std::ostream& os, float    v) { os.write(reinterpret_cast<const char*>(&v), 4); }
inline void write_f64(std::ostream& os, double   v) { os.write(reinterpret_cast<const char*>(&v), 8); }

inline void write_str(std::ostream& os, const std::string& s) {
    write_u64(os, s.size());
    if (!s.empty()) os.write(s.data(), static_cast<std::streamsize>(s.size()));
}

inline uint8_t  read_u8 (std::istream& is) { uint8_t  v; is.read(reinterpret_cast<char*>(&v), 1); return v; }
inline uint32_t read_u32(std::istream& is) { uint32_t v; is.read(reinterpret_cast<char*>(&v), 4); return v; }
inline uint64_t read_u64(std::istream& is) { uint64_t v; is.read(reinterpret_cast<char*>(&v), 8); return v; }
inline int32_t  read_i32(std::istream& is) { int32_t  v; is.read(reinterpret_cast<char*>(&v), 4); return v; }
inline float    read_f32(std::istream& is) { float    v; is.read(reinterpret_cast<char*>(&v), 4); return v; }
inline double   read_f64(std::istream& is) { double   v; is.read(reinterpret_cast<char*>(&v), 8); return v; }

inline std::string read_str(std::istream& is) {
    uint64_t len = read_u64(is);
    std::string s(static_cast<std::size_t>(len), '\0');
    if (len > 0) is.read(s.data(), static_cast<std::streamsize>(len));
    return s;
}

}  // namespace trifecta::io
