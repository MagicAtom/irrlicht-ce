//
//  IH264FileParser.h
//  Irrlicht_OSX
//
//  Created by 宋庭聿 on 2021/11/23.
//

#ifndef __IRR_I_H264_FILE_PARSER_H
#define __IRR_I_H264_FILE_PARSER_H

#include <optional>
#include "IStream.h"
#ifdef _WIN32
#include <winsock2.h>
#else
#include <arpa/inet.h>
#endif

namespace irr{
namespace video{

class H264FileParser : public StreamSource{
public:
    H264FileParser(uint32_t fps, bool loop):sampleDuration_us(1000*1000/fps),StreamSource() {
        
    }
    void start() override{
        sampleTime_us = std::numeric_limits<uint64_t>::max() - sampleDuration_us + 1;
    };
    void stop() override{
        StreamSource::stop();
    };
    const uint64_t sampleDuration_us;
    
    void loadNextSample(std::vector<uint8_t>& buffer){
        sampleTime_us += sampleDuration_us;
        sample = *reinterpret_cast<std::vector<std::byte>*>(&buffer);
    };
    void clearSentSample(){
        sample.clear();
        sample.shrink_to_fit();
    };
};

}
}


#endif /* IH264FileParser_h */
