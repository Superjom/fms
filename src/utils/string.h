#ifndef _FMS_UTILS_STRING_H_
#define _FMS_UTILS_STRING_H_

namespace fms {
// count 开头连续空格
inline size_t count_spaces(const char* s) {
    size_t count = 0;
    while (*s != 0 && isspace(*s++)) {
        count++;
    }
    return count;
}
std::vector<std::string> split(const std::string &line, const std::string &pattern) {
    std::vector<std::string> result;
    std::string::size_type pos = 0, found = 0;
    while(found != std::string::npos) {
        found = line.find(pattern, pos);
        result.push_back(line.substr(pos, found - pos));
        pos = found + 1;
    }
    return result;
}

std::string trim(const std::string &str)
{
    std::string tags = " \n\t\r";
    std::string::size_type pos = str.find_first_not_of(tags);
    if(pos == std::string::npos)
    {
        return str;
    }
    std::string::size_type pos2 = str.find_last_not_of(tags);
    if(pos2 != std::string::npos)
    {
        return std::move(str.substr(pos, pos2 - pos + 1));
    }
    return std::move(str.substr(pos));
}


}; // end namespace fms


#endif
