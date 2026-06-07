#pragma once

#include "imgui.h"

#include <algorithm>
#include <cmath>

namespace RS::UI {

inline float Clamp01(float v) {
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

inline void ApplyReferenceStyle() {
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowPadding     = ImVec2(18.0f, 16.0f);
    style.FramePadding      = ImVec2(12.0f, 8.0f);
    style.CellPadding       = ImVec2(10.0f, 7.0f);
    style.ItemSpacing       = ImVec2(10.0f, 10.0f);
    style.ItemInnerSpacing  = ImVec2(8.0f, 8.0f);
    style.IndentSpacing     = 16.0f;
    style.ScrollbarSize     = 12.0f;
    style.GrabMinSize       = 22.0f;

    style.WindowBorderSize  = 1.0f;
    style.ChildBorderSize   = 1.0f;
    style.PopupBorderSize   = 1.0f;
    style.FrameBorderSize   = 0.0f;
    style.TabBorderSize     = 0.0f;

    style.WindowRounding    = 18.0f;
    style.ChildRounding     = 16.0f;
    style.PopupRounding     = 14.0f;
    style.FrameRounding     = 13.0f;
    style.GrabRounding      = 12.0f;
    style.ScrollbarRounding = 10.0f;
    style.TabRounding       = 10.0f;

    style.WindowTitleAlign  = ImVec2(0.02f, 0.50f);
    style.ColorButtonPosition = ImGuiDir_Right;

    ImVec4* c = style.Colors;
    c[ImGuiCol_Text]                  = ImVec4(0.88f, 0.88f, 0.86f, 1.00f);
    c[ImGuiCol_TextDisabled]          = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
    c[ImGuiCol_WindowBg]              = ImVec4(0.045f, 0.045f, 0.045f, 0.96f);
    c[ImGuiCol_ChildBg]               = ImVec4(0.065f, 0.065f, 0.065f, 0.95f);
    c[ImGuiCol_PopupBg]               = ImVec4(0.070f, 0.070f, 0.070f, 0.98f);
    c[ImGuiCol_Border]                = ImVec4(0.17f, 0.17f, 0.17f, 1.00f);
    c[ImGuiCol_BorderShadow]          = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    c[ImGuiCol_FrameBg]               = ImVec4(0.120f, 0.120f, 0.120f, 1.00f);
    c[ImGuiCol_FrameBgHovered]        = ImVec4(0.170f, 0.170f, 0.170f, 1.00f);
    c[ImGuiCol_FrameBgActive]         = ImVec4(0.210f, 0.210f, 0.210f, 1.00f);
    c[ImGuiCol_TitleBg]               = ImVec4(0.055f, 0.055f, 0.055f, 1.00f);
    c[ImGuiCol_TitleBgActive]         = ImVec4(0.075f, 0.075f, 0.075f, 1.00f);
    c[ImGuiCol_TitleBgCollapsed]      = ImVec4(0.045f, 0.045f, 0.045f, 0.92f);
    c[ImGuiCol_MenuBarBg]             = ImVec4(0.060f, 0.060f, 0.060f, 1.00f);
    c[ImGuiCol_ScrollbarBg]           = ImVec4(0.050f, 0.050f, 0.050f, 0.80f);
    c[ImGuiCol_ScrollbarGrab]         = ImVec4(0.22f, 0.22f, 0.22f, 1.00f);
    c[ImGuiCol_ScrollbarGrabHovered]  = ImVec4(0.32f, 0.32f, 0.32f, 1.00f);
    c[ImGuiCol_ScrollbarGrabActive]   = ImVec4(0.44f, 0.44f, 0.44f, 1.00f);
    c[ImGuiCol_CheckMark]             = ImVec4(0.92f, 0.92f, 0.90f, 1.00f);
    c[ImGuiCol_SliderGrab]            = ImVec4(0.82f, 0.82f, 0.80f, 1.00f);
    c[ImGuiCol_SliderGrabActive]      = ImVec4(0.96f, 0.96f, 0.94f, 1.00f);
    c[ImGuiCol_Button]                = ImVec4(0.120f, 0.120f, 0.120f, 1.00f);
    c[ImGuiCol_ButtonHovered]         = ImVec4(0.180f, 0.180f, 0.180f, 1.00f);
    c[ImGuiCol_ButtonActive]          = ImVec4(0.240f, 0.240f, 0.240f, 1.00f);
    c[ImGuiCol_Header]                = ImVec4(0.140f, 0.140f, 0.140f, 1.00f);
    c[ImGuiCol_HeaderHovered]         = ImVec4(0.210f, 0.210f, 0.210f, 1.00f);
    c[ImGuiCol_HeaderActive]          = ImVec4(0.280f, 0.280f, 0.280f, 1.00f);
    c[ImGuiCol_Separator]             = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);
    c[ImGuiCol_SeparatorHovered]      = ImVec4(0.42f, 0.42f, 0.40f, 1.00f);
    c[ImGuiCol_SeparatorActive]       = ImVec4(0.68f, 0.68f, 0.64f, 1.00f);
    c[ImGuiCol_ResizeGrip]            = ImVec4(0.36f, 0.36f, 0.34f, 0.25f);
    c[ImGuiCol_ResizeGripHovered]     = ImVec4(0.58f, 0.58f, 0.54f, 0.55f);
    c[ImGuiCol_ResizeGripActive]      = ImVec4(0.84f, 0.84f, 0.78f, 0.80f);
    c[ImGuiCol_Tab]                   = ImVec4(0.090f, 0.090f, 0.090f, 1.00f);
    c[ImGuiCol_TabHovered]            = ImVec4(0.190f, 0.190f, 0.190f, 1.00f);
    c[ImGuiCol_TabActive]             = ImVec4(0.145f, 0.145f, 0.145f, 1.00f);
    c[ImGuiCol_TabUnfocused]          = ImVec4(0.070f, 0.070f, 0.070f, 1.00f);
    c[ImGuiCol_TabUnfocusedActive]    = ImVec4(0.115f, 0.115f, 0.115f, 1.00f);
    c[ImGuiCol_DockingPreview]        = ImVec4(0.84f, 0.84f, 0.78f, 0.38f);
    c[ImGuiCol_DockingEmptyBg]        = ImVec4(0.030f, 0.030f, 0.030f, 0.00f);
    c[ImGuiCol_TableHeaderBg]         = ImVec4(0.095f, 0.095f, 0.095f, 1.00f);
    c[ImGuiCol_TableBorderStrong]     = ImVec4(0.22f, 0.22f, 0.22f, 1.00f);
    c[ImGuiCol_TableBorderLight]      = ImVec4(0.15f, 0.15f, 0.15f, 1.00f);
    c[ImGuiCol_TableRowBg]            = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    c[ImGuiCol_TableRowBgAlt]         = ImVec4(1.00f, 1.00f, 1.00f, 0.035f);
    c[ImGuiCol_TextSelectedBg]        = ImVec4(0.64f, 0.64f, 0.60f, 0.28f);
    c[ImGuiCol_DragDropTarget]        = ImVec4(0.90f, 0.90f, 0.82f, 0.90f);
    c[ImGuiCol_NavHighlight]          = ImVec4(0.78f, 0.78f, 0.70f, 0.60f);
}

inline void BeginDockspace() {
#ifdef IMGUI_HAS_DOCK
    ImGuiDockNodeFlags flags = ImGuiDockNodeFlags_PassthruCentralNode;
    ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), flags);
#endif
}

inline float DragSpeed(float minValue, float maxValue) {
    const float span = maxValue - minValue;
    if (span > 0.0f && std::isfinite(span)) {
        return std::max(span / 240.0f, 0.00001f);
    }
    return 0.01f;
}

inline bool Toggle(const char* label, bool* value) {
    const ImGuiStyle& style = ImGui::GetStyle();
    const float startX = ImGui::GetCursorPosX();
    const float rowY = ImGui::GetCursorPosY();
    const float avail = ImGui::GetContentRegionAvail().x;
    const float height = ImGui::GetFrameHeight();
    const float width = height * 1.85f;

    ImGui::PushID(label);
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(label);

    ImGui::SameLine();
    ImGui::SetCursorPosX(startX + std::max(0.0f, avail - width));
    ImGui::SetCursorPosY(rowY);

    bool changed = false;
    if (ImGui::InvisibleButton("##toggle", ImVec2(width, height))) {
        *value = !*value;
        changed = true;
    }

    const bool hovered = ImGui::IsItemHovered();
    const ImVec2 min = ImGui::GetItemRectMin();
    const ImVec2 max = ImGui::GetItemRectMax();
    const float radius = height * 0.5f;
    const float knobRadius = radius - 3.0f;
    const float knobX = *value ? (max.x - radius) : (min.x + radius);
    const float knobY = (min.y + max.y) * 0.5f;

    const ImVec4 trackOn  = hovered ? ImVec4(0.66f, 0.66f, 0.64f, 1.0f)
                                    : ImVec4(0.54f, 0.54f, 0.52f, 1.0f);
    const ImVec4 trackOff = hovered ? ImVec4(0.20f, 0.20f, 0.20f, 1.0f)
                                    : ImVec4(0.13f, 0.13f, 0.13f, 1.0f);
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(min, max, ImGui::GetColorU32(*value ? trackOn : trackOff), radius);
    draw->AddCircleFilled(ImVec2(knobX, knobY), knobRadius,
                          ImGui::GetColorU32(ImVec4(0.91f, 0.91f, 0.89f, 1.0f)), 24);

    ImGui::PopID();
    return changed;
}

inline bool SliderFloat(const char* label, float* value, float minValue,
                        float maxValue, const char* format = "%.3f") {
    const ImGuiStyle& style = ImGui::GetStyle();
    const float startX = ImGui::GetCursorPosX();
    const float rowY = ImGui::GetCursorPosY();
    const float avail = ImGui::GetContentRegionAvail().x;
    const float labelWidth = std::min(160.0f, std::max(96.0f, avail * 0.34f));
    const float valueWidth = std::min(112.0f, std::max(76.0f, avail * 0.24f));
    const float sliderWidth = avail - labelWidth - valueWidth - style.ItemInnerSpacing.x * 2.0f;

    ImGui::PushID(label);
    bool changed = false;
    if (sliderWidth >= 84.0f) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(label);
        ImGui::SameLine();
        ImGui::SetCursorPosX(startX + labelWidth);
        ImGui::SetCursorPosY(rowY);
        ImGui::SetNextItemWidth(valueWidth);
        changed |= ImGui::DragFloat("##value", value, DragSpeed(minValue, maxValue),
                                    minValue, maxValue, format);
        ImGui::SameLine(0.0f, style.ItemInnerSpacing.x);
        ImGui::SetNextItemWidth(sliderWidth);
        changed |= ImGui::SliderFloat("##track", value, minValue, maxValue, "");
    } else {
        ImGui::TextUnformatted(label);
        ImGui::SetNextItemWidth(valueWidth);
        changed |= ImGui::DragFloat("##value", value, DragSpeed(minValue, maxValue),
                                    minValue, maxValue, format);
        ImGui::SameLine(0.0f, style.ItemInnerSpacing.x);
        ImGui::SetNextItemWidth(std::max(80.0f, avail - valueWidth - style.ItemInnerSpacing.x));
        changed |= ImGui::SliderFloat("##track", value, minValue, maxValue, "");
    }
    ImGui::PopID();
    return changed;
}

inline bool SliderInt(const char* label, int* value, int minValue, int maxValue) {
    const ImGuiStyle& style = ImGui::GetStyle();
    const float startX = ImGui::GetCursorPosX();
    const float rowY = ImGui::GetCursorPosY();
    const float avail = ImGui::GetContentRegionAvail().x;
    const float labelWidth = std::min(160.0f, std::max(96.0f, avail * 0.34f));
    const float valueWidth = std::min(92.0f, std::max(64.0f, avail * 0.20f));
    const float sliderWidth = avail - labelWidth - valueWidth - style.ItemInnerSpacing.x * 2.0f;

    ImGui::PushID(label);
    bool changed = false;
    if (sliderWidth >= 84.0f) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(label);
        ImGui::SameLine();
        ImGui::SetCursorPosX(startX + labelWidth);
        ImGui::SetCursorPosY(rowY);
        ImGui::SetNextItemWidth(valueWidth);
        changed |= ImGui::DragInt("##value", value, 1.0f, minValue, maxValue);
        ImGui::SameLine(0.0f, style.ItemInnerSpacing.x);
        ImGui::SetNextItemWidth(sliderWidth);
        changed |= ImGui::SliderInt("##track", value, minValue, maxValue, "");
    } else {
        ImGui::TextUnformatted(label);
        ImGui::SetNextItemWidth(valueWidth);
        changed |= ImGui::DragInt("##value", value, 1.0f, minValue, maxValue);
        ImGui::SameLine(0.0f, style.ItemInnerSpacing.x);
        ImGui::SetNextItemWidth(std::max(80.0f, avail - valueWidth - style.ItemInnerSpacing.x));
        changed |= ImGui::SliderInt("##track", value, minValue, maxValue, "");
    }
    ImGui::PopID();
    return changed;
}

inline bool SliderFloatRows(const char* label, float* values, int count,
                            float minValue, float maxValue,
                            const char* format = "%.3f") {
    static const char* kAxisLabels[] = { "X", "Y", "Z", "W" };
    ImGui::TextUnformatted(label);
    ImGui::PushID(label);
    ImGui::Indent(10.0f);
    bool changed = false;
    for (int i = 0; i < count; ++i) {
        changed |= SliderFloat(kAxisLabels[i], &values[i], minValue, maxValue, format);
    }
    ImGui::Unindent(10.0f);
    ImGui::PopID();
    return changed;
}

inline bool SliderFloat2Rows(const char* label, float* values, float minValue,
                             float maxValue, const char* format = "%.3f") {
    return SliderFloatRows(label, values, 2, minValue, maxValue, format);
}

inline bool SliderFloat3Rows(const char* label, float* values, float minValue,
                             float maxValue, const char* format = "%.3f") {
    return SliderFloatRows(label, values, 3, minValue, maxValue, format);
}

} // namespace RS::UI
