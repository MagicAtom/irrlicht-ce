//
//  IRtcHelpers.h
//  Irrlicht_OSX
//
//  Created by 宋庭聿 on 2021/11/23.
//

#ifndef __IRR_I_RTC_HELPERS_H
#define __IRR_I_RTC_HELPERS_H

#include "rtc/rtc.hpp"
#include <shared_mutex>
#include <ctime>
#include <iostream>
namespace irr{
namespace video{
#ifdef _WIN32
// taken from https://stackoverflow.com/questions/10905892/equivalent-of-gettimeday-for-windows
#include <windows.h>
#include <winsock2.h> // for struct timeval

struct timezone {
    int tz_minuteswest;
    int tz_dsttime;
};

static int gettimeofday(struct timeval *tv, struct timezone *tz)
{
    if (tv) {
        FILETIME               filetime; /* 64-bit value representing the number of 100-nanosecond intervals since January 1, 1601 00:00 UTC */
        ULARGE_INTEGER         x;
        ULONGLONG              usec;
        static const ULONGLONG epoch_offset_us = 11644473600000000ULL; /* microseconds betweeen Jan 1,1601 and Jan 1,1970 */

#if _WIN32_WINNT >= _WIN32_WINNT_WIN8
        GetSystemTimePreciseAsFileTime(&filetime);
#else
        GetSystemTimeAsFileTime(&filetime);
#endif
        x.LowPart =  filetime.dwLowDateTime;
        x.HighPart = filetime.dwHighDateTime;
        usec = x.QuadPart / 10  -  epoch_offset_us;
        tv->tv_sec  = (time_t)(usec / 1000000ULL);
        tv->tv_usec = (long)(usec % 1000000ULL);
    }
    if (tz) {
        TIME_ZONE_INFORMATION timezone;
        GetTimeZoneInformation(&timezone);
        tz->tz_minuteswest = timezone.Bias;
        tz->tz_dsttime = 0;
    }
    return 0;
}
#else
#include <sys/time.h>
#endif

struct ClientTrackData {
    std::shared_ptr<rtc::Track> track;
    std::shared_ptr<rtc::RtcpSrReporter> sender;

    ClientTrackData(std::shared_ptr<rtc::Track> track, std::shared_ptr<rtc::RtcpSrReporter> sender){
        this->track = track;
        this->sender = sender;
    }
};

struct Client {
    enum class State {
        Waiting,
        WaitingForVideo,
        Ready
    };
    const std::shared_ptr<rtc::PeerConnection> & peerConnection = _peerConnection;
    Client(std::shared_ptr<rtc::PeerConnection> pc) {
        _peerConnection = pc;
    }
    std::optional<std::shared_ptr<ClientTrackData>> video;
    std::optional<std::shared_ptr<rtc::DataChannel>> dataChannel{};
    void setState(State state){
        std::unique_lock lock(_mutex);
        this->state = state;
    }
    State getState(){
        std::shared_lock lock(_mutex);
        return state;
    }

private:
    std::shared_mutex _mutex;
    State state = State::Waiting;
    std::string id;
    std::shared_ptr<rtc::PeerConnection> _peerConnection;
};

struct ClientTrack {
    std::string id;
    std::shared_ptr<ClientTrackData> trackData;
    ClientTrack(std::string id, std::shared_ptr<ClientTrackData> trackData){
        this->id = id;
        this->trackData = trackData;
    }
};

}
}


#endif /* IRtcHelpers_h */
