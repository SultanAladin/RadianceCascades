// Source/Core/Logger.h
// Tiny print-to-stderr logger so subsystems don't pull in fmt or spdlog.
// Macros keep the call sites cheap; the variadic forwards to vfprintf.
#pragma once

namespace RS {

void LogInfo (const char* fmt, ...);
void LogWarn (const char* fmt, ...);
void LogError(const char* fmt, ...);

} // namespace RS

#define RS_LOG_INFO(...)  ::RS::LogInfo (__VA_ARGS__)
#define RS_LOG_WARN(...)  ::RS::LogWarn (__VA_ARGS__)
#define RS_LOG_ERROR(...) ::RS::LogError(__VA_ARGS__)
