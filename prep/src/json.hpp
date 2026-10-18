#pragma once
// Minimal JSON value, parser and writer: leaves in/out and golden parity files. Objects keep insertion order.
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace prep::json {

struct Value;
using Array = std::vector<Value>;
using Object = std::vector<std::pair<std::string, Value>>;

struct Value {
    // Integers stay integers (Python json writes 1, not 1.0; consumers index arrays with them).
    std::variant<std::nullptr_t, bool, double, std::string, std::shared_ptr<Array>, std::shared_ptr<Object>, int64_t> v = nullptr;
    Value() = default;
    Value(std::nullptr_t) {}
    Value(bool b) : v(b) {}
    Value(double d) : v(d) {}
    Value(int i) : v(int64_t(i)) {}
    Value(int64_t i) : v(i) {}
    Value(size_t i) : v(int64_t(i)) {}
    Value(const char* s) : v(std::string(s)) {}
    Value(std::string s) : v(std::move(s)) {}
    Value(Array a) : v(std::make_shared<Array>(std::move(a))) {}
    Value(Object o) : v(std::make_shared<Object>(std::move(o))) {}
    static Value array() { return Value(Array{}); }
    static Value object() { return Value(Object{}); }

    bool isNull() const { return std::holds_alternative<std::nullptr_t>(v); }
    bool isNumber() const { return std::holds_alternative<double>(v) || std::holds_alternative<int64_t>(v); }
    bool isArray() const { return std::holds_alternative<std::shared_ptr<Array>>(v); }
    bool isObject() const { return std::holds_alternative<std::shared_ptr<Object>>(v); }
    bool isString() const { return std::holds_alternative<std::string>(v); }
    double num() const {
        if (auto b = std::get_if<bool>(&v)) return *b;
        if (auto i = std::get_if<int64_t>(&v)) return double(*i);
        return std::get<double>(v);
    }
    bool boolean() const { return std::get<bool>(v); }
    const std::string& str() const { return std::get<std::string>(v); }
    Array& arr() { return *std::get<std::shared_ptr<Array>>(v); }
    const Array& arr() const { return *std::get<std::shared_ptr<Array>>(v); }
    Object& obj() { return *std::get<std::shared_ptr<Object>>(v); }
    const Object& obj() const { return *std::get<std::shared_ptr<Object>>(v); }
    size_t size() const { return isArray() ? arr().size() : obj().size(); }
    const Value& operator[](size_t i) const { return arr()[i]; }
    const Value* find(const std::string& key) const {
        if (!isObject()) return nullptr;
        for (auto& [k, value] : obj())
            if (k == key) return &value;
        return nullptr;
    }
    const Value& operator[](const std::string& key) const {
        auto p = find(key);
        if (!p) throw std::runtime_error("json: missing key " + key);
        return *p;
    }
    Value& set(const std::string& key, Value value) {
        for (auto& [k, existing] : obj())
            if (k == key) return existing = std::move(value);
        obj().emplace_back(key, std::move(value));
        return obj().back().second;
    }
    void push(Value value) { arr().push_back(std::move(value)); }
};

class Parser {
    const std::string& s;
    size_t i = 0;
    void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) ++i; }
    [[noreturn]] void fail(const char* what) { throw std::runtime_error(std::string("json parse: ") + what + " at " + std::to_string(i)); }
    std::string string() {
        std::string out;
        ++i;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\') {
                ++i;
                switch (s[i]) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    unsigned code = std::stoul(s.substr(i + 1, 4), nullptr, 16);
                    i += 4;
                    if (code < 0x80) out += char(code);
                    else if (code < 0x800) { out += char(0xc0 | (code >> 6)); out += char(0x80 | (code & 0x3f)); }
                    else { out += char(0xe0 | (code >> 12)); out += char(0x80 | ((code >> 6) & 0x3f)); out += char(0x80 | (code & 0x3f)); }
                    break;
                }
                default: out += s[i];
                }
                ++i;
            } else out += s[i++];
        }
        if (i >= s.size()) fail("unterminated string");
        ++i;
        return out;
    }

public:
    explicit Parser(const std::string& text) : s(text) {}
    Value value() {
        ws();
        if (i >= s.size()) fail("unexpected end");
        char c = s[i];
        if (c == '{') {
            ++i;
            Value out = Value::object();
            ws();
            if (s[i] == '}') { ++i; return out; }
            while (true) {
                ws();
                if (s[i] != '"') fail("expected key");
                std::string key = string();
                ws();
                if (s[i++] != ':') fail("expected :");
                out.obj().emplace_back(std::move(key), value());
                ws();
                if (s[i] == ',') { ++i; continue; }
                if (s[i] == '}') { ++i; return out; }
                fail("expected , or }");
            }
        }
        if (c == '[') {
            ++i;
            Value out = Value::array();
            ws();
            if (s[i] == ']') { ++i; return out; }
            while (true) {
                out.arr().push_back(value());
                ws();
                if (s[i] == ',') { ++i; continue; }
                if (s[i] == ']') { ++i; return out; }
                fail("expected , or ]");
            }
        }
        if (c == '"') return Value(string());
        if (s.compare(i, 4, "true") == 0) { i += 4; return Value(true); }
        if (s.compare(i, 5, "false") == 0) { i += 5; return Value(false); }
        if (s.compare(i, 4, "null") == 0) { i += 4; return Value(); }
        if (s.compare(i, 3, "NaN") == 0) { i += 3; return Value(std::nan("")); }
        if (s.compare(i, 8, "Infinity") == 0) { i += 8; return Value(INFINITY); }
        if (s.compare(i, 9, "-Infinity") == 0) { i += 9; return Value(-INFINITY); }
        const char* begin = s.data() + i;
        const char* last = s.data() + s.size();
        double d = 0;
        auto [end, error] = std::from_chars(begin, last, d);
        if (error != std::errc() || end == begin) fail("bad number");
        bool integer = std::string_view(begin, size_t(end - begin)).find_first_of(".eE") == std::string_view::npos;
        i += size_t(end - begin);
        if (integer) {
            int64_t n = 0;
            std::from_chars(begin, end, n);
            return Value(n);
        }
        return Value(d);
    }
};

inline Value parse(const std::string& text) { return Parser(text).value(); }

inline void number(std::string& out, double d) {
    if (std::isnan(d)) { out += "NaN"; return; }
    if (std::isinf(d)) { out += d > 0 ? "Infinity" : "-Infinity"; return; }
    // Shortest representation that round-trips (like Python repr); integral floats keep a ".0" so they stay floats.
    char buffer[40];
    auto [end, error] = std::to_chars(buffer, buffer + sizeof buffer, d);
    std::string_view text(buffer, size_t(end - buffer));
    out += text;
    if (text.find_first_of(".eE") == std::string_view::npos) out += ".0";
}

inline void dump(const Value& value, std::string& out, int indent = -1, int depth = 0) {
    auto newline = [&](int d) {
        if (indent < 0) return;
        out += '\n';
        out.append(size_t(indent * d), ' ');
    };
    if (value.isNull()) out += "null";
    else if (auto b = std::get_if<bool>(&value.v)) out += *b ? "true" : "false";
    else if (auto n = std::get_if<int64_t>(&value.v)) out += std::to_string(*n);
    else if (value.isNumber()) number(out, value.num());
    else if (value.isString()) {
        out += '"';
        for (char c : value.str()) {
            if (c == '"' || c == '\\') { out += '\\'; out += c; }
            else if (c == '\n') out += "\\n";
            else out += c;
        }
        out += '"';
    } else if (value.isArray()) {
        auto& a = value.arr();
        // Numeric leaf arrays stay on one line: mesh grids would otherwise be one number per line.
        bool flat = std::all_of(a.begin(), a.end(), [](const Value& x) { return !x.isArray() && !x.isObject(); });
        out += '[';
        for (size_t k = 0; k < a.size(); ++k) {
            if (k) out += flat || indent < 0 ? ", " : ",";
            if (!flat) newline(depth + 1);
            dump(a[k], out, flat ? -1 : indent, depth + 1);
        }
        if (!flat && !a.empty()) newline(depth);
        out += ']';
    } else {
        auto& o = value.obj();
        out += '{';
        for (size_t k = 0; k < o.size(); ++k) {
            if (k) out += ',';
            newline(depth + 1);
            dump(Value(o[k].first), out);
            out += indent < 0 ? ":" : ": ";
            dump(o[k].second, out, indent, depth + 1);
        }
        if (!o.empty()) newline(depth);
        out += '}';
    }
}

inline std::string dump(const Value& value, int indent = -1) {
    std::string out;
    dump(value, out, indent);
    return out;
}

} // namespace prep::json
