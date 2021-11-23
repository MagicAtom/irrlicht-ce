//
//  IDispatchQueue.h
//  Irrlicht_OSX
//
//  Created by 宋庭聿 on 2021/11/23.
//

#ifndef __IRR_I_DISPATCH_QUEUE_H
#define __IRR_I_DISPATCH_QUEUE_H

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <iostream>

namespace irr{
namespace video{

class DispatchQueue {
    typedef std::function<void(void)> fp_t;

public:
    DispatchQueue(std::string name, size_t threadCount = 1): name{std::move(name)}, threads(threadCount) {
        for(size_t i = 0; i < threads.size(); i++)
        {
            threads[i] = std::thread(&DispatchQueue::dispatchThreadHandler, this);
        }
    }
    ~DispatchQueue(){
        std::unique_lock<std::mutex> lock(lockMutex);
        quit = true;
        lock.unlock();
        condition.notify_all();

        // Wait for threads to finish before we exit
        for(size_t i = 0; i < threads.size(); i++)
        {
            if(threads[i].joinable())
            {
                threads[i].join();
            }
        }
    };

    // dispatch and copy
    void dispatch(const fp_t& op){
        std::unique_lock<std::mutex> lock(lockMutex);
            queue.push(op);

            // Manual unlocking is done before notifying, to avoid waking up
            // the waiting thread only to block again (see notify_one for details)
            lock.unlock();
            condition.notify_one();
    };
    // dispatch and move
    void dispatch(fp_t&& op){
        std::unique_lock<std::mutex> lock(lockMutex);
            queue.push(std::move(op));

            // Manual unlocking is done before notifying, to avoid waking up
            // the waiting thread only to block again (see notify_one for details)
            lock.unlock();
            condition.notify_one();
    };

    void removePending(){
        std::unique_lock<std::mutex> lock(lockMutex);
            queue = {};
    };

    // Deleted operations
    DispatchQueue(const DispatchQueue& rhs) = delete;
    DispatchQueue& operator=(const DispatchQueue& rhs) = delete;
    DispatchQueue(DispatchQueue&& rhs) = delete;
    DispatchQueue& operator=(DispatchQueue&& rhs) = delete;

private:
    std::string name;
    std::mutex lockMutex;
    std::vector<std::thread> threads;
    std::queue<fp_t> queue;
    std::condition_variable condition;
    bool quit = false;

    void dispatchThreadHandler(void){
        std::unique_lock<std::mutex> lock(lockMutex);
            do {
                //Wait until we have data or a quit signal
                condition.wait(lock, [this]{
                    return (queue.size() || quit);
                });

                //after wait, we own the lock
                if(!quit && queue.size())
                {
                    auto op = std::move(queue.front());
                    queue.pop();

                    //unlock now that we're done messing with the queue
                    lock.unlock();

                    op();

                    lock.lock();
                }
            } while (!quit);
    }
};

}
}

#endif /* IDispatchQueue_h */
