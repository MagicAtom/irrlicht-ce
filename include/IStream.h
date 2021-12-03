//
//  IStream.h
//  Irrlicht_OSX
//
//  Created by 宋庭聿 on 2021/11/23.
//

#ifndef __IRR_I_STREAM_H
#define __IRR_I_STREAM_H

#include "IDispatchQueue.h"
#include "rtc/rtc.hpp"
#include "IRtcHelpers.h"
#include <ctime>

#ifdef _WIN32
// taken from https://stackoverflow.com/questions/5801813/c-usleep-is-obsolete-workarounds-for-windows-mingw
#include <windows.h>

void usleep(__int64 usec)
{
    HANDLE timer;
    LARGE_INTEGER ft;

    ft.QuadPart = -(10*usec); // Convert to 100 nanosecond interval, negative value indicates relative time

    timer = CreateWaitableTimer(NULL, TRUE, NULL);
    SetWaitableTimer(timer, &ft, 0, NULL, NULL, 0);
    WaitForSingleObject(timer, INFINITE);
    CloseHandle(timer);
}
#else
#include <unistd.h>
#endif

namespace irr{
namespace video{

class StreamSource {
protected:
    uint64_t sampleTime_us = 0;
    rtc::binary sample = {};

public:
    StreamSource() { }
    virtual void start() = 0;
    virtual void stop(){
        sampleTime_us = 0;
        sample = {};
    };
    virtual void loadNextSample(std::vector<uint8_t>& buffer) = 0;

    inline uint64_t getSampleTime_us() { return sampleTime_us; }
    inline rtc::binary getSample() { return sample; }

    ~StreamSource(){
        stop();
    };
};

class Stream: std::enable_shared_from_this<Stream> {
    uint64_t startTime = 0;
    std::mutex mutex;
    DispatchQueue dispatchQueue = DispatchQueue("StreamQueue");

    bool _isRunning = false;
public:
    const std::shared_ptr<StreamSource> video;
    Stream(std::shared_ptr<StreamSource> video): std::enable_shared_from_this<Stream>(), video(video) { }
    enum class StreamSourceType {
        Audio,
        Video
    };
    ~Stream(){
        stop();
    };

private:
    rtc::synchronized_callback<StreamSourceType, uint64_t, rtc::binary> sampleHandler;

    void sendSample();
public:
    void publishSample(){
        if(!isRunning)
                return;
        auto ss = video;
        auto sst = StreamSourceType::Video;
        auto sample = ss->getSample();
        sampleHandler(sst,ss->getSampleTime_us(),sample);
    };
    void onSample(std::function<void (StreamSourceType, uint64_t, rtc::binary)> handler){
        sampleHandler = handler;
    };
    void start(){
        std::lock_guard lock(mutex);
        if (isRunning) {
            return;
        }
        _isRunning = true;
        auto currentTimeInMicroSeconds = [](){
                struct timeval time;
                gettimeofday(&time, NULL);
                return uint64_t(time.tv_sec) * 1000 * 1000 + time.tv_usec;
        };
        startTime = currentTimeInMicroSeconds();
        video->start();
    };
    void stop(){
        std::lock_guard lock(mutex);
            if (!isRunning) {
                return;
            }
            _isRunning = false;
            dispatchQueue.removePending();
            //audio->stop();
            video->stop();
    };
    const bool & isRunning = _isRunning;
};

}
}

#endif /* IStream_h */
