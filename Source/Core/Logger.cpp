// Source/Core/Logger.cpp
#include "Core/Logger.h"

#include <cstdarg>
#include <cstdio>

namespace RS {

static void Vlog(const char* prefix, const char* fmt, va_list args) {
    std::fprintf(stderr, "%s", prefix);
    std::vfprintf(stderr, fmt, args);
    std::fputc('\n', stderr);
}

void LogInfo(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    Vlog("[RS][INFO] ", fmt, args);
    va_end(args);
}

void LogWarn(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    Vlog("[RS][WARN] ", fmt, args);
    va_end(args);
}

void LogError(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    Vlog("[RS][ERROR] ", fmt, args);
    va_end(args);
}

} // namespace RS
