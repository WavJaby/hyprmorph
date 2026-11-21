#include "types.hpp"

namespace prep {

json::Value toJson(const Grid& grid) {
    json::Array rows;
    for (int r = 0; r < grid.rows; ++r) {
        json::Array row;
        for (int c = 0; c < grid.cols; ++c) row.push_back(json::Array{grid.x(r, c), grid.y(r, c)});
        rows.push_back(std::move(row));
    }
    return rows;
}

Grid gridFromJson(const json::Value& value) {
    Grid grid(int(value.size()), int(value[0].size()));
    for (int r = 0; r < grid.rows; ++r)
        for (int c = 0; c < grid.cols; ++c) {
            grid.x(r, c) = value[size_t(r)][size_t(c)][0].num();
            grid.y(r, c) = value[size_t(r)][size_t(c)][1].num();
        }
    return grid;
}

static json::Value box(const Box& b) { return json::Array{b[0], b[1], b[2], b[3]}; }
static json::Value box(const BoxF& b) { return json::Array{b[0], b[1], b[2], b[3]}; }

json::Value toJson(const Leaf& leaf) {
    json::Value out = json::Value::object();
    out.set("name", leaf.name);
    out.set("kind", leaf.kind);
    out.set("old_box", box(leaf.oldBox));
    out.set("new_box", box(leaf.newBox));
    out.set("midpoint_box", box(leaf.midpointBox));
    if (leaf.clip) {
        json::Array anchor;
        for (Anchor a : *leaf.clip) anchor.push_back(a == Anchor::End ? "end" : "start");
        out.set("clip_anchor", anchor);
    }
    if (leaf.meshes) {
        json::Value meshes = json::Value::object();
        meshes.set("old_source", toJson(leaf.meshes->oldSource));
        meshes.set("old_midpoint", toJson(leaf.meshes->oldMidpoint));
        meshes.set("new_source", toJson(leaf.meshes->newSource));
        meshes.set("new_midpoint", toJson(leaf.meshes->newMidpoint));
        out.set("local_meshes", meshes);
        out.set("local_mesh_coordinate_convention", "pixel centers within region patch");
    }
    if (leaf.rigid) {
        json::Value rigid = *leaf.rigid;
        json::Array windows;
        for (auto& w : leaf.windows) windows.push_back(json::Array{w[0], w[1], w[2], w[3]});
        rigid.set("fixed_windows_vertex_indices", windows);
        out.set("rigid_constraints", rigid);
    }
    return out;
}

Leaf leafFromJson(const json::Value& value) {
    Leaf leaf;
    leaf.name = value["name"].str();
    leaf.kind = value["kind"].str();
    for (int k = 0; k < 4; ++k) {
        leaf.oldBox[size_t(k)] = int(value["old_box"][size_t(k)].num());
        leaf.newBox[size_t(k)] = int(value["new_box"][size_t(k)].num());
        leaf.midpointBox[size_t(k)] = value["midpoint_box"][size_t(k)].num();
    }
    if (auto anchor = value.find("clip_anchor"))
        leaf.clip = std::array<Anchor, 2>{(*anchor)[0].str() == "end" ? Anchor::End : Anchor::Start, (*anchor)[1].str() == "end" ? Anchor::End : Anchor::Start};
    if (auto meshes = value.find("local_meshes"); meshes && meshes->isObject() && meshes->size()) {
        Meshes m;
        m.oldSource = gridFromJson((*meshes)["old_source"]);
        m.oldMidpoint = gridFromJson((*meshes)["old_midpoint"]);
        m.newSource = gridFromJson((*meshes)["new_source"]);
        m.newMidpoint = gridFromJson((*meshes)["new_midpoint"]);
        leaf.meshes = std::move(m);
    }
    if (auto rigid = value.find("rigid_constraints")) {
        leaf.rigid = *rigid;
        if (auto windows = rigid->find("fixed_windows_vertex_indices"))
            for (auto& w : windows->arr())
                leaf.windows.push_back({int(w[0].num()), int(w[1].num()), int(w[2].num()), int(w[3].num())});
    }
    return leaf;
}

json::Value toJson(const std::vector<Leaf>& leaves) {
    json::Array out;
    for (auto& leaf : leaves) out.push_back(toJson(leaf));
    return out;
}

std::vector<Leaf> leavesFromJson(const json::Value& value) {
    std::vector<Leaf> leaves;
    for (auto& item : value.arr()) leaves.push_back(leafFromJson(item));
    return leaves;
}

} // namespace prep
