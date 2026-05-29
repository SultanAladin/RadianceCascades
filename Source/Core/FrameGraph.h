// Source/Core/FrameGraph.h
// Linear list-of-passes orchestrator with manual barriers. Kept intentionally
// small in v1; an auto-barrier rewrite is v2 work.
#pragma once

namespace RS {

class FrameGraph {
public:
    FrameGraph();
    ~FrameGraph();
};

} // namespace RS
